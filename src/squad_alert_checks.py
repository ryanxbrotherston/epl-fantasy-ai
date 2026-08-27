"""
squad_alert_checks.py — shared injury/suspension/confirmed-lineup checking
logic for squad alert monitors.

Extracted out of ai_team_monitor.py so the same checks (official FPL status,
official confirmed team sheet via highlightly.net, and the early bench-likely
signal from fpledits.com) can be reused by more than one monitor - the AI
team's own watchdog (ai_team_monitor.py) and, per DECISION_friend_alerts.md,
friend_alert_monitor.py, which runs the identical checks against each
opted-in friend's own squad. Nothing here is specific to either caller; both
just supply their own picks_df/all_players/bootstrap and handle the results
(dedup log, email recipient, etc.) themselves.
"""

from datetime import datetime, timezone

import pandas as pd

import confirmed_lineup
import fpl_api
import lineup_predictor

CONFIRMED_LOOKUP_WINDOW_HOURS = 2  # only spend Highlightly API calls this close to a kickoff -
                                    # its lineups aren't released until ~30-40min before anyway,
                                    # and its free tier is 100 req/day, not enough to poll hourly
                                    # all week for fixtures nowhere near kickoff yet

UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}
DOUBTFUL_THRESHOLD = 50  # chance_of_playing_next_round % AT OR BELOW which a "doubtful" player
                          # still gets flagged - "50% or less" per conversation 2026-08-27, hence
                          # the <= comparison in flag_problem_players() below (not <)


def deadline_has_passed(gw: dict) -> bool:
    """True once `gw`'s (fpl_api.current_gameweek()'s return value) transfer deadline is in the
    past. fpl_api.current_gameweek() keeps returning the SAME event (is_current=True) for the
    entire rest of that gameweek, well past its own deadline - individual fixtures kick off at
    different times all week, but the one deadline that actually locks transfers/lineup changes
    already passed. Both monitors' alert emails suggest a change to make "before the deadline
    above" - past that point the suggestion is no longer actionable, so callers should skip
    alerting entirely rather than keep emailing hourly for the rest of the gameweek (see
    conversation 2026-08-27: exactly this caused an hourly inbox-clogging alert loop)."""
    deadline = datetime.fromisoformat(gw["deadline_time"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc) >= deadline


def flag_problem_players(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Squad members who are injured/suspended/unavailable, or doubtful below
    threshold, and are in the starting XI (squad_slot <= 11, i.e. not benched -
    NOT multiplier > 0, confirmed live 2026-08-27 that a past/finished
    gameweek's picks can carry multiplier=1 for every one of the 15
    including bench, so that's not a reliable signal there)."""
    starting = picks_df[picks_df["squad_slot"] <= 11].copy()
    hard_out = starting["status"].isin(UNAVAILABLE_STATUSES)
    chance = pd.to_numeric(starting.get("chance_of_playing_next_round"), errors="coerce")
    doubtful = starting["status"].eq("d") & chance.notna() & (chance <= DOUBTFUL_THRESHOLD)
    return starting[hard_out | doubtful]


def flag_bench_likely_players(picks_df: pd.DataFrame, predicted_starting_ids: set | None,
                               already_flagged_ids: set) -> pd.DataFrame:
    """Starting XI players who look bench-likely per the EARLY signal
    (fpledits.com's predicted lineup) but aren't already covered by a more
    authoritative check (official FPL status, or an official confirmed
    team sheet) - i.e. genuinely new information, not a duplicate of
    either. Returns an empty frame if the early source was unreachable
    this run (predicted_starting_ids is None) - a fetch failure must
    never get silently treated as "everyone's benched"."""
    starting = picks_df[picks_df["squad_slot"] <= 11].copy()
    if predicted_starting_ids is None or starting.empty:
        return starting.iloc[0:0]

    remaining = starting[~starting["element"].isin(already_flagged_ids)].copy()
    if remaining.empty:
        return remaining

    flags = remaining.apply(
        lambda r: lineup_predictor.starting_likelihood_flag(
            r["element"], r["status"], predicted_starting_ids,
            chance_of_playing=pd.to_numeric(r.get("chance_of_playing_next_round"), errors="coerce"),
            news=r.get("news") or None,  # FPL uses "" for "no news", not null - normalize to None
        ),
        axis=1,
    )
    remaining["early_flag"] = [f["flag"] for f in flags]
    return remaining[remaining["early_flag"] == "red"]


def flag_confirmed_benched_players(picks_df: pd.DataFrame, all_players: pd.DataFrame, bootstrap: dict,
                                    target_event: int, already_flagged_ids: set) -> pd.DataFrame:
    """Starting XI players whose OFFICIAL team sheet (via highlightly.net,
    see confirmed_lineup.py) has genuinely been released and confirms them
    NOT starting - not a prediction, the real thing. Only ever checks
    fixtures within CONFIRMED_LOOKUP_WINDOW_HOURS of kickoff (rate-limit
    conscious - see module header) - for anyone else, returns nothing for
    them rather than guessing, exactly like an unreached/not-yet-released
    source does. Excludes anyone already flagged (official status takes
    priority; a name-match failure here must never override it either way)."""
    starting = picks_df[picks_df["squad_slot"] <= 11].copy()
    starting = starting[~starting["element"].isin(already_flagged_ids)]
    if starting.empty:
        return starting

    team_id_to_name = {t["id"]: t["name"] for t in bootstrap["teams"]}
    names_by_id = all_players.set_index("id")[["first_name", "second_name"]]
    fixtures = fpl_api.get_fixtures(target_event)
    now = datetime.now(timezone.utc)

    flagged_rows = []
    lineup_cache = {}  # highlightly_match_id -> confirmed lineup dict, avoid re-fetching per player
    for team_id in starting["team"].unique():
        fixture = next((f for f in fixtures if f["team_h"] == team_id or f["team_a"] == team_id), None)
        if fixture is None or fixture.get("kickoff_time") is None:
            continue
        kickoff = datetime.fromisoformat(fixture["kickoff_time"].replace("Z", "+00:00"))
        hours_to_kickoff = (kickoff - now).total_seconds() / 3600
        if not (0 <= hours_to_kickoff <= CONFIRMED_LOOKUP_WINDOW_HOURS or -0.5 <= hours_to_kickoff < 0):
            continue  # not close enough to kickoff yet (or long finished) - don't spend the call

        home_name = team_id_to_name[fixture["team_h"]]
        away_name = team_id_to_name[fixture["team_a"]]
        match_id = confirmed_lineup.find_match_id(home_name, away_name, fixture["kickoff_time"])
        if match_id is None:
            continue

        if match_id not in lineup_cache:
            lineup_cache[match_id] = confirmed_lineup.get_confirmed_lineup_names(match_id)
        confirmed = lineup_cache[match_id]
        if confirmed is None:
            continue  # not released yet, or Highlightly unreachable - no data, not "not started"

        team_players = starting[starting["team"] == team_id]
        for _, player in team_players.iterrows():
            if player["element"] not in names_by_id.index:
                continue
            first, second = names_by_id.loc[player["element"], ["first_name", "second_name"]]
            status = confirmed_lineup.player_confirmed_status(first, second, player["web_name"], confirmed)
            if status is False:
                flagged_rows.append(player)

    return pd.DataFrame(flagged_rows) if flagged_rows else starting.iloc[0:0]


def _eligible_candidates(problem_player: pd.Series, squad_ids: set, bank: int,
                          all_players: pd.DataFrame, team_limit_counts: dict) -> pd.DataFrame:
    """Same-position, budget-feasible, club-limit-respecting, available candidates
    for replacing problem_player - the shared filtering both suggest_replacement()
    (single best pick, for email alert bodies) and suggest_replacements() (ranked
    top-N, for the My Team UI) build on, so the eligibility rules can never drift
    between the two."""
    max_price = problem_player["now_cost"] + bank
    pos = problem_player["position"]
    team_of_departing = problem_player.get("team")

    candidates = all_players[
        (all_players["position"] == pos)
        & (all_players["now_cost"] <= max_price)
        & (~all_players["id"].isin(squad_ids))
        & (~all_players["status"].isin(UNAVAILABLE_STATUSES))
    ].copy()

    def club_ok(row):
        count = team_limit_counts.get(row["team"], 0)
        if row["team"] == team_of_departing:
            count -= 1  # departing player frees up a club slot
        return count < 3

    return candidates[candidates.apply(club_ok, axis=1)]


def suggest_replacement(problem_player: pd.Series, squad_ids: set, bank: int,
                         all_players: pd.DataFrame, team_limit_counts: dict) -> pd.Series | None:
    """Best same-position, budget-feasible, club-limit-respecting replacement,
    ranked by FPL's own ep_next blended with points_per_game (the live model's
    rolling component isn't recomputed here - this is a simpler in-season
    proxy, see README)."""
    candidates = _eligible_candidates(problem_player, squad_ids, bank, all_players, team_limit_counts)
    if candidates.empty:
        return None

    ep_next = pd.to_numeric(candidates["ep_next"], errors="coerce").fillna(0)
    ppg = pd.to_numeric(candidates["points_per_game"], errors="coerce").fillna(0)
    candidates["score"] = 0.6 * ep_next + 0.4 * ppg
    return candidates.sort_values("score", ascending=False).iloc[0]


def suggest_replacements(problem_player: pd.Series, squad_ids: set, bank: int,
                          preds_all: pd.DataFrame, team_limit_counts: dict,
                          top_n: int = 10) -> pd.DataFrame:
    """Top-N same-position, budget/club-limit-feasible replacements, ranked by
    our own model's predicted_points (the "overall strength" column - the same
    blended model/ep_next/ppg score the transfer planner itself optimizes on,
    NOT the simpler ep_next+ppg proxy suggest_replacement() above uses for a
    one-line email suggestion). preds_all must be the full model-scored table
    (app.py's refresh_predictions_with_live_data() output), not the plain FPL
    bootstrap frame - only that one carries predicted_points and roll_total_points.
    Callers get season performance (points_per_game) and historic performance
    (roll_total_points - despite the name, this is a 5-gameweek ROLLING AVERAGE
    from the end of last season, per build_gw1_features.py's ROLLING_WINDOW=5,
    not a season total; 0 if this player has no matched prior-season history)
    as separate columns
    alongside the ranking, rather than folding everything into one opaque score -
    the ranking axis should be legible, not just a number to trust blindly."""
    candidates = _eligible_candidates(problem_player, squad_ids, bank, preds_all, team_limit_counts)
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("predicted_points", ascending=False).head(top_n).copy()
    if "roll_total_points" not in candidates.columns:
        candidates["roll_total_points"] = 0.0
    candidates["roll_total_points"] = pd.to_numeric(candidates["roll_total_points"], errors="coerce").fillna(0.0)
    return candidates


def rank_squad_trades(squad_df: pd.DataFrame, squad_ids: set, bank: int, preds_all: pd.DataFrame,
                       team_limit_counts: dict, top_n: int = 10) -> pd.DataFrame:
    """The top N single-swap trade IDEAS across the WHOLE squad - not a list of
    alternatives for one already-flagged problem player. For each of the 15 squad
    members, finds their single best same-position/budget/club-limit-feasible
    replacement (via suggest_replacements(), keeping only its #1 pick), then ranks
    ALL those (out, in) pairs together by net predicted_points gain. Each row is a
    complete, nameable trade - "OUT: X, IN: Y" - not just a pool of incoming
    candidates (see conversation 2026-08-27: this replaced an earlier version that
    only ranked replacements for one flagged player, which missed the point).

    squad_df must be the 15-man squad's rows from preds_all (not all_players) - same
    reason as suggest_replacements() above, it needs predicted_points/roll_total_points.

    A near-zero or negative top gain is a REAL, VALID answer, not a bug - FPL lets you
    bank up to 5 free transfers, so "none of these are worth it, save the transfer" is
    a legitimate recommendation this ranking should be able to produce, not something
    to hide by forcing a trade to look good."""
    rows = []
    for _, out_player in squad_df.iterrows():
        picks = suggest_replacements(out_player, squad_ids, bank, preds_all, team_limit_counts, top_n=1)
        if picks.empty:
            continue
        in_player = picks.iloc[0]
        out_points = float(pd.to_numeric(pd.Series([out_player.get("predicted_points", 0)]),
                                          errors="coerce").fillna(0).iloc[0])
        rows.append({
            "out_id": out_player["id"], "out_name": out_player["web_name"],
            "out_points": out_points,
            "in_id": in_player["id"], "in_name": in_player["web_name"],
            "in_team": in_player["team_name"], "in_cost": in_player["now_cost"],
            "in_points": in_player["predicted_points"],
            "in_ppg": in_player.get("points_per_game"),
            "in_roll_total_points": in_player.get("roll_total_points", 0.0),
            "gain": in_player["predicted_points"] - out_points,
        })
    if not rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values("gain", ascending=False).head(top_n).reset_index(drop=True)
