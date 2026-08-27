"""
app.py — EPL Fantasy AI dashboard

Three tabs:
  AI Team    - the model's optimized squad, fully automated, no human input
  My Team    - your own Team ID: pulls your real squad, flags issues
  Friends    - anyone's Team ID: same view, shareable once deployed

Run with:  streamlit run app.py
"""

import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import fpl_api
import lineup_predictor
import manager_store
import confirmed_lineup
import team_visuals
import pitch_view
import squad_alert_checks
import fixture_ticker
import price_predictor
import risk_analyzer
import multi_week_planner
import multi_week_projections
import player_props
from squad_optimizer import load_predictions, pick_squad, pick_starting_xi, BUDGET

MODEL_DIR = Path(__file__).parent / "models"

CONFIRMED_LOOKUP_WINDOW_HOURS = 3  # a bit more generous than ai_team_monitor.py's 2h -
                                    # this is checked on-demand by page loads, not a fixed
                                    # hourly cron, so a wider net avoids missing a release
                                    # between visits; the 10min cache below bounds the cost

st.set_page_config(page_title="EPL Fantasy AI", page_icon="⚽", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

# Theme colors/font live in .streamlit/config.toml (Streamlit's own [theme]
# section) - this block only adds what config.toml can't do declaratively:
# real :hover states, and the active-tab fill. Streamlit's tab internals
# aren't a documented/stable public API - they already broke once (was
# button[data-baseweb="tab"], silently stopped matching after a version
# bump). Currently div[data-testid="stTab"] with role="tab" and
# aria-selected="true"/"false", confirmed by inspecting the deployed app's
# actual DOM. If tabs stop highlighting again after a future Streamlit
# upgrade, re-inspect the live DOM rather than guess a selector - this is
# the first place to check.
# Dark broadcast-sports-graphics palette (see .streamlit/config.toml's own
# comment for the rationale/limits of taking this only as far as native
# Streamlit chrome can go - the custom-rendered pitch in pitch_view.py is
# where the real quality bar is met). Tokens mirror pitch_view.py's own
# :root custom properties so the two don't drift - duplicated rather than
# shared because they're injected via two separate st.markdown calls with
# no common stylesheet.
st.markdown("""
<style>
:root {
    --app-accent: #6C5CE8; --app-accent-soft: rgba(108,92,232,0.16);
    --app-panel-elevated: #171A28; --app-border: rgba(255,255,255,0.08);
    --app-text-primary: #F2F3F7;
}
div[data-testid="stTab"] {
    transition: background-color 0.15s ease, color 0.15s ease;
    border-radius: 8px 8px 0 0;
    cursor: pointer;
}
div[data-testid="stTab"]:hover {
    background-color: var(--app-accent-soft) !important;
}
div[data-testid="stTab"]:hover p {
    color: var(--app-text-primary) !important;
}
div[data-testid="stTab"][aria-selected="true"] {
    background-color: var(--app-accent-soft) !important;
}
div[data-testid="stTab"][aria-selected="true"] p {
    color: var(--app-text-primary) !important;
}
div[data-testid="stMetric"] {
    background: var(--app-panel-elevated);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    border: 1px solid var(--app-border);
}
div[data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
}
</style>
""", unsafe_allow_html=True)


# ---------- cached data loaders ----------

@st.cache_data(ttl=3600)
def load_bootstrap():
    return fpl_api.get_bootstrap_static()


@st.cache_data(ttl=3600)
def load_base_predictions():
    """The pre-trained model's blended predictions from our seed pipeline
    (run offline via src/build_gw1_features.py + src/squad_optimizer.py)."""
    return load_predictions()


@st.cache_data(ttl=900)
def load_predicted_starting_ids():
    """The EARLY-prediction signal (fpledits.com's predicted lineups) -
    15min cache since this is refreshed through the week as team news
    develops, unlike the hourly-cached bootstrap/predictions above."""
    return lineup_predictor.fetch_predicted_starting_ids()


@st.cache_data(ttl=600)
def load_confirmed_lineups_for_gameweek(_bootstrap: dict, target_event: int) -> dict:
    """{team_id: {'starters': set[name], 'bench': set[name]}} for every
    club whose fixture this gameweek is within CONFIRMED_LOOKUP_WINDOW_HOURS
    of kickoff AND has a genuinely released highlightly.net lineup -
    everyone else just isn't in this dict (same "no data yet" contract as
    confirmed_lineup.py itself). 10min cache, shared across every visitor
    hitting this deployed app - keeps this bounded well under
    Highlightly's 100 req/day free tier regardless of traffic, same
    principle as the kickoff-window gating in ai_team_monitor.py.
    Leading underscore on _bootstrap tells st.cache_data not to hash it
    (it's large and already itself cached/stable within its own TTL)."""
    from datetime import datetime, timezone
    team_id_to_name = {t["id"]: t["name"] for t in _bootstrap["teams"]}
    fixtures = fpl_api.get_fixtures(target_event)
    now = datetime.now(timezone.utc)
    result = {}
    for fixture in fixtures:
        if fixture.get("kickoff_time") is None:
            continue
        kickoff = datetime.fromisoformat(fixture["kickoff_time"].replace("Z", "+00:00"))
        hours_to_kickoff = (kickoff - now).total_seconds() / 3600
        if not (-0.5 <= hours_to_kickoff <= CONFIRMED_LOOKUP_WINDOW_HOURS):
            continue
        home_name = team_id_to_name[fixture["team_h"]]
        away_name = team_id_to_name[fixture["team_a"]]
        match_id = confirmed_lineup.find_match_id(home_name, away_name, fixture["kickoff_time"])
        if match_id is None:
            continue
        confirmed = confirmed_lineup.get_confirmed_lineup_names(match_id)
        if confirmed is None:
            continue
        result[fixture["team_h"]] = confirmed
        result[fixture["team_a"]] = confirmed
    return result


@st.cache_resource
def load_match_model_bundle() -> dict:
    """match_model.pkl - the fitted attack/defence model fixture_ticker.py and
    multi_week_projections.py both build on. A model object, not tabular
    data, so st.cache_resource (not cache_data) is the right fit - same
    reasoning as manager_store.get_client()'s own use of it."""
    return joblib.load(MODEL_DIR / "match_model.pkl")


@st.cache_data(ttl=3600)
def load_league_avg_lambda(_bundle: dict, team_names: tuple) -> float:
    """The fixture-difficulty baseline every ticker/projection call needs -
    iterates every team/opponent/venue combination, not cheap, so this is
    cached for the same hour-long window as bootstrap itself (team list is
    season-stable). Leading underscore on _bundle: same "don't hash this,
    it's already stable" convention as _bootstrap above."""
    return player_props.league_average_lambda(_bundle, list(team_names))


@st.cache_data(ttl=3600)
def load_fixtures_df(_bootstrap: dict) -> pd.DataFrame:
    """All of this season's fixtures as a DataFrame with the team_h_name/
    team_a_name columns fixture_ticker.build_ticker() and
    multi_week_projections.build_projection_table() both expect - FPL's raw
    fixtures/ endpoint only gives team_h/team_a as numeric ids."""
    team_id_to_name = {t["id"]: t["name"] for t in _bootstrap["teams"]}
    fixtures = fpl_api.get_fixtures()
    df = pd.DataFrame(fixtures)
    df["team_h_name"] = df["team_h"].map(team_id_to_name)
    df["team_a_name"] = df["team_a"].map(team_id_to_name)
    return df


@st.cache_data(ttl=60)
def load_event_live(event: int) -> dict | None:
    """Short TTL relative to everything else here - this is the one piece
    of genuinely live, time-sensitive data (B6's provisional score), unlike
    the hourly-stable bootstrap/predictions above."""
    return fpl_api.get_event_live(event)


_BASIS_LABEL = {
    "confirmed": "official status", "confirmed_lineup": "confirmed team sheet",
    "early": "predicted", "none": "no data",
}


def format_rank(entry: dict) -> str:
    """entry['summary_overall_rank'] - FPL's own last-known overall rank
    field (confirmed against a live response 2026-08-20). Null pre-season
    and until a manager's first gameweek is scored - not a bug, that's
    genuinely FPL's own state right now."""
    rank = entry.get("summary_overall_rank")
    return f"{rank:,}" if rank is not None else "N/A yet"


def compute_starting_likelihood(row: pd.Series, id_col: str, predicted_ids: set | None,
                                 confirmed_lineups: dict, names_by_id: pd.DataFrame) -> dict:
    """The single shared per-player flag computation - both the list-view
    'Starting?' column and the pitch view's flag indicators are built
    from this, so the two can never drift apart. Returns
    lineup_predictor.starting_likelihood_flag()'s raw dict."""
    player_id = row[id_col]
    confirmed_status = None
    team_id = row.get("team")
    confirmed = confirmed_lineups.get(team_id) if team_id is not None else None
    if confirmed is not None and player_id in names_by_id.index:
        first, second = names_by_id.loc[player_id, ["first_name", "second_name"]]
        confirmed_status = confirmed_lineup.player_confirmed_status(
            first, second, row.get("web_name", ""), confirmed
        )
    chance = pd.to_numeric(row.get("chance_of_playing_next_round"), errors="coerce")
    news = row.get("news") or None  # FPL uses "" for "no news", not null - normalize to None
    return lineup_predictor.starting_likelihood_flag(
        player_id, row["status"], predicted_ids, confirmed_status, chance, news,
    )


def build_pitch_cards(df: pd.DataFrame, id_col: str, points_col: str, bootstrap: dict,
                       all_players: pd.DataFrame, target_event: int,
                       captain_id=None, vice_id=None) -> list[dict]:
    """Normalizes any of the three tabs' differently-shaped squad
    dataframes into pitch_view's plain card-dict list - the one place
    that shape translation happens, so pitch_view.py itself never needs
    to know about fpl_api's column names vs. squad_optimizer's."""
    predicted_ids = load_predicted_starting_ids()
    confirmed_lineups = load_confirmed_lineups_for_gameweek(bootstrap, target_event)
    names_by_id = all_players.set_index("id")[["first_name", "second_name"]]
    team_code_by_id = {t["id"]: t["code"] for t in bootstrap["teams"]}

    cards = []
    for _, row in df.iterrows():
        result = compute_starting_likelihood(row, id_col, predicted_ids, confirmed_lineups, names_by_id)
        points = pd.to_numeric(pd.Series([row.get(points_col)]), errors="coerce").iloc[0]
        points = 0.0 if pd.isna(points) else float(points)
        cards.append(pitch_view.build_card(
            id_=row[id_col], name=row["web_name"], position=row["position"],
            team_code=team_code_by_id.get(row["team"], -1), points=points,
            is_captain=row[id_col] == captain_id, is_vice=row[id_col] == vice_id,
            flag_result={"flag": result["flag"], "basis": _BASIS_LABEL[result["basis"]], "detail": result["detail"]},
        ))
    return cards


def render_squad_pitch(xi_df: pd.DataFrame, bench_df: pd.DataFrame, id_col: str, points_col: str,
                        bootstrap: dict, all_players: pd.DataFrame, target_event: int,
                        captain_id=None, vice_id=None) -> tuple[list[dict], list[dict]]:
    """The one call site every tab uses to go from a squad's XI/bench
    dataframes to a rendered pitch - formation is derived from whatever's
    actually in xi_df, not assumed. Returns (xi_cards, bench_cards) so
    callers needing the same flag/basis/detail this pitch already computed
    per starter (e.g. My Team's suggested-changes section) can reuse it
    directly instead of re-deriving it - see render_suggested_changes()."""
    xi_cards = build_pitch_cards(xi_df, id_col, points_col, bootstrap, all_players, target_event,
                                  captain_id, vice_id)
    bench_cards = build_pitch_cards(bench_df, id_col, points_col, bootstrap, all_players, target_event,
                                     captain_id, vice_id)
    st.caption(f"Formation: {pitch_view.formation_from_xi(xi_cards)}")
    pitch_html = pitch_view.PITCH_CSS + pitch_view.render_pitch(xi_cards, bench_cards)
    # Collapse every whitespace run (including newlines) to a single space
    # before handing this to st.markdown. Two separate problems, confirmed
    # live in this session's own local run, not hypothetical:
    #  1. render_pitch()/PITCH_CSS build their HTML with normal Python
    #     source indentation, and Markdown's CommonMark spec treats a line
    #     indented 4+ spaces (after a blank line) as a literal code block -
    #     left as-is, the whole pitch renders as a wall of visible HTML text.
    #  2. A first pass that only stripped LEADING whitespace per line (still
    #     leaving real newlines inside multi-line tags, e.g. the flag dot's
    #     onclick attribute wrapped onto its own line) broke worse: whatever
    #     markdown-to-HTML step st.markdown runs the content through doesn't
    #     treat a tag split across lines as one element - it silently
    #     dropped the onclick/other wrapped attributes and even changed a
    #     <span> into a <p> in one case. Fully collapsing to single-line
    #     tags avoids that class of corruption entirely. HTML itself doesn't
    #     care about whitespace between/within tags, so this is a no-op for
    #     the actual layout either way.
    pitch_html = re.sub(r"\s+", " ", pitch_html)
    st.markdown(pitch_html, unsafe_allow_html=True)
    return xi_cards, bench_cards


PPG_TRUST_RAMP_STARTS = 5  # squad_optimizer.load_predictions()'s "ppg" component is documented
                            # as a "proven quality anchor" - true pre-season, when it's still last
                            # season's points-per-game. Once refresh_predictions_with_live_data()
                            # below swaps in bootstrap-static's LIVE points_per_game, that anchor
                            # becomes this season's mean so far - with only 1-2 games played, that's
                            # literally "how many points did you just score", not a quality signal,
                            # and it swings the 20%-weighted blend hard off one good/bad game (e.g.
                            # flagging a season-long starter for transfer-out after one poor game,
                            # or a bench player for transfer-in after one good one - see conversation
                            # 2026-08-27). Below this many starts this season, ppg's weight is
                            # linearly shrunk toward 0 and handed to the stable season-long model
                            # component instead; by PPG_TRUST_RAMP_STARTS it's back to full trust.


def refresh_predictions_with_live_data(base_preds: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    """Prices, injury status, and FPL's own ep_next drift daily pre-season -
    pull those fresh from bootstrap-static rather than trusting the snapshot
    the model was seeded from, while keeping our trained model's point
    estimate (that part doesn't go stale day to day)."""
    live = fpl_api.players_dataframe(bootstrap)
    live_cols = live[["id", "now_cost", "status", "ep_next", "points_per_game",
                       "chance_of_playing_next_round", "team_name", "news", "starts"]]

    merged = base_preds.drop(
        columns=["now_cost", "status", "ep_next", "points_per_game",
                 "chance_of_playing_next_round", "team_name", "news"],
        errors="ignore",
    ).merge(live_cols, on="id", how="left")

    from squad_optimizer import BLEND_WEIGHTS, UNAVAILABLE_STATUSES, apply_price_bias_correction
    ep_next = pd.to_numeric(merged["ep_next"], errors="coerce").fillna(merged["model_points"])
    ppg = pd.to_numeric(merged["points_per_game"], errors="coerce").fillna(merged["model_points"])

    starts = pd.to_numeric(merged["starts"], errors="coerce").fillna(0)
    ppg_trust = (starts / PPG_TRUST_RAMP_STARTS).clip(upper=1.0)
    ppg_weight = BLEND_WEIGHTS["ppg"] * ppg_trust
    model_weight = BLEND_WEIGHTS["model"] + (BLEND_WEIGHTS["ppg"] - ppg_weight)

    merged["predicted_points"] = (
        model_weight * merged["model_points"]
        + BLEND_WEIGHTS["ep_next"] * ep_next
        + ppg_weight * ppg
    )
    chance = pd.to_numeric(merged["chance_of_playing_next_round"], errors="coerce")
    doubtful = merged["status"].eq("d") & chance.notna()
    merged.loc[doubtful, "predicted_points"] *= (chance[doubtful] / 100)

    unavailable = merged["status"].isin(UNAVAILABLE_STATUSES)
    merged["available"] = ~unavailable

    merged = apply_price_bias_correction(merged)

    # Genuinely unavailable (injured/suspended/left club) players were previously left with
    # their full blended predicted_points (plus whatever the price-bias correction just added
    # on top) - the model/ep_next/ppg blend has no idea they can't play at all, so a squad
    # member out injured could still show a positive score and get silently kept (or even
    # started) by the same solver that's supposed to be deciding whether to transfer them out.
    # Zeroing this out AFTER the price correction (not before - it'd just get partially undone)
    # is what actually makes the transfer planner (and its hit_confidence_margin bar - see
    # typical_gameweek_score()) correctly weigh "this squad is carrying N injuries, each
    # currently worth 0" as the real, large points swing it is, rather than needing a separate
    # injury-count heuristic bolted on top.
    merged.loc[unavailable, "predicted_points"] = 0.0
    return merged


# ---------- sidebar: gameweek status ----------

st.title("⚽ EPL Fantasy AI")
st.caption("The dot on each player's badge = starting likelihood (tap it for detail). "
           "🟢🟡🔴 with **official status** = FPL's own status field. With **confirmed team sheet** = "
           "highlightly.net's real released lineup (only exists ~30-40min pre-kickoff). With "
           "**predicted** = fpledits.com's third-party guess, refreshed through the week - not official.")

try:
    bootstrap = load_bootstrap()
    gw = fpl_api.current_gameweek(bootstrap)
    st.sidebar.metric("Next deadline", gw["name"])
    st.sidebar.caption(gw["deadline_time"].replace("T", " ").replace("Z", " UTC"))
    all_players = fpl_api.players_dataframe(bootstrap)
    live_ok = True
except Exception as e:
    st.sidebar.error(f"Couldn't reach the live FPL API: {e}")
    st.sidebar.caption("Falling back to the offline seed data from the last build.")
    bootstrap = None
    all_players = None
    live_ok = False


tab_ai, tab_mine, tab_friends = st.tabs(["🤖 AI Team", "👤 My Team", "👥 Friends"])


# ---------- AI TEAM ----------

def get_ai_team_id():
    """AI_TEAM_ID is only set once you've actually created the AI-run FPL
    account and want the site to switch from 'recommendation mode' to
    'live read-only display of the real team'. Configure it in Streamlit
    Cloud's secrets (not committed to the repo) or as an env var locally."""
    try:
        return int(st.secrets["AI_TEAM_ID"])
    except Exception:
        import os
        val = os.environ.get("AI_TEAM_ID")
        return int(val) if val else None


# ---------- shared: team lookup by ID ----------

def render_team_lookup(key_prefix: str, default_id: int | None = None, show_suggested_changes: bool = False):
    """show_suggested_changes gates every piece of stats-backed ADVICE below
    the pitch (suggested changes, transfer advice, best-XI recommendation,
    season history, the differentials watchlist) - only My Team's two call
    sites pass True. Friends is purely for looking up someone else's team,
    not giving them advice (nobody but the account owner should be told what
    to do with someone else's squad), so Friends and the AI Team tab both
    get the plain lookup/pitch with nothing below it."""
    team_id = st.number_input("FPL Team ID", min_value=1, step=1, value=default_id, key=f"{key_prefix}_id",
                               help="Find this in the FPL site URL when viewing 'Pick Team' or 'Gameweek history'.")
    if st.button("Load team", key=f"{key_prefix}_load"):
        st.session_state[f"{key_prefix}_loaded"] = True
    if not st.session_state.get(f"{key_prefix}_loaded"):
        return

    if not live_ok:
        st.error("Live FPL API isn't reachable right now, so I can't pull a real team. Try again shortly.")
        return

    entry = fpl_api.get_entry(int(team_id))
    if entry is None:
        st.error(f"No FPL team found with ID {team_id} — double check the number.")
        return

    st.success(f"**{entry['name']}** — managed by {entry['player_first_name']} {entry['player_last_name']}")

    current_event = gw["id"] if gw["is_current"] else max(gw["id"] - 1, 1)
    result = fpl_api.team_picks_dataframe(int(team_id), current_event, bootstrap)

    if result is None:
        st.info(f"No squad submitted yet for Gameweek {current_event}, or picks aren't public until "
                f"the deadline passes. Check back after **{gw['deadline_time']}**.")
        return

    picks_df, history = result
    st.markdown(f"##### Gameweek {current_event} squad")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total points", entry.get("summary_overall_points") or "N/A yet")
    c2.metric("Overall rank", format_rank(entry))
    c3.metric("Bank", f"£{history['bank'] / 10:.1f}m")
    c4.metric("Team value", f"£{history['value'] / 10:.1f}m")
    c5.metric("Free transfers used", history.get("event_transfers", 0))

    if show_suggested_changes:
        render_live_score(picks_df, current_event)  # B6 - most time-sensitive, so it goes near the top

    xi = picks_df[picks_df["multiplier"] > 0]
    bench = picks_df[picks_df["multiplier"] == 0]
    captain_row = picks_df[picks_df["is_captain"]]
    vice_row = picks_df[picks_df["is_vice_captain"]]
    xi_cards, bench_cards = render_squad_pitch(
        xi, bench, "element", "ep_next", bootstrap, all_players, current_event,
        captain_id=captain_row["element"].iloc[0] if not captain_row.empty else None,
        vice_id=vice_row["element"].iloc[0] if not vice_row.empty else None,
    )

    if not show_suggested_changes:
        return

    squad_ids = set(picks_df["element"])
    team_limit_counts = all_players[all_players["id"].isin(squad_ids)]["team"].value_counts().to_dict()
    bank = history.get("bank", 0)

    render_suggested_changes(xi, xi_cards, all_players, bank, squad_ids, team_limit_counts)  # B2
    render_transfer_advice(key_prefix, int(team_id), picks_df, history, current_event)       # B3
    render_best_xi_recommendation(picks_df)                                                  # B4
    render_season_history(int(team_id))                                                      # B5
    render_watchlist()                                                                        # B7


# ---------- My Team only: B6, live provisional score ----------

def render_live_score(picks_df: pd.DataFrame, current_event: int):
    """Only while a gameweek is actually IN PROGRESS - confirmed live
    (2026-08-21) that bootstrap-static's events carry is_current/is_next/
    is_previous/finished/data_checked booleans; "live right now" is
    is_current and not finished. Deliberately does NOT attempt FPL's own
    automatic-substitution algorithm (bench order + position eligibility +
    only resolves once minutes are final) - that's real, non-trivial logic
    out of scope for this pass. Sums the starting XI's live points with each
    pick's own 'multiplier' (1 normal, 2 captain, 3 triple-captain chip) -
    more robust than hardcoding "captain doubled", since it already handles
    the triple-captain case for free."""
    event = next((e for e in bootstrap["events"] if e["id"] == current_event), None)
    if event is None or not event["is_current"] or event["finished"]:
        return  # not live right now - no stale/zeroed section between gameweeks

    live = load_event_live(current_event)
    if not live or not live.get("elements"):
        return  # gameweek is "current" but matches haven't actually kicked off yet

    live_points = {e["id"]: e.get("stats", {}).get("total_points", 0) for e in live["elements"]}
    xi = picks_df[picks_df["multiplier"] > 0]
    total = sum(live_points.get(row["element"], 0) * row["multiplier"] for _, row in xi.iterrows())

    st.metric("Live score (provisional)", total)
    st.caption("Provisional - doesn't yet account for automatic substitutions (bench order, "
               "position eligibility, and final minutes aren't resolved here).")


# ---------- My Team only: B2, stats-backed suggested changes ----------

def render_suggested_changes(xi: pd.DataFrame, xi_cards: list[dict], all_players: pd.DataFrame,
                              bank: int, squad_ids: set, team_limit_counts: dict):
    """Reuses each starting-XI card's already-computed flag (xi_cards, built
    once by render_squad_pitch/build_pitch_cards) rather than re-deriving
    it - the same dot color the pitch already shows. Confirmed (red) and
    early/predicted (yellow) are grouped separately, mirroring the CONFIRMED
    vs EARLY WARNING split ai_team_monitor.py's own email alerts already
    use, for the same reason: a released team sheet and a third-party guess
    carry very different confidence."""
    xi_reset = xi.reset_index(drop=True)
    pairs = list(zip((row for _, row in xi_reset.iterrows()), xi_cards))
    red = [p for p in pairs if p[1]["flag"] == "red"]
    yellow = [p for p in pairs if p[1]["flag"] == "yellow"]
    if not red and not yellow:
        return  # nothing flagged - don't render an empty section

    st.markdown("##### Suggested changes")

    team_id_to_name = {t["id"]: t["name"] for t in bootstrap["teams"]}
    bundle = load_match_model_bundle()
    avg_lambda = load_league_avg_lambda(bundle, tuple(t["name"] for t in bootstrap["teams"]))
    ticker = fixture_ticker.build_ticker(load_fixtures_df(bootstrap), bundle, avg_lambda, n_gameweeks=3)
    fixture_summary = fixture_ticker.team_summary(ticker) if not ticker.empty else pd.DataFrame()

    price_moves = price_predictor.predict_price_moves(all_players, bootstrap["total_players"])
    price_by_id, rise_cutoff, fall_cutoff = None, None, None
    if not price_moves.empty:
        price_by_id = price_moves.set_index("id")
        rise_cutoff = price_moves["prob_rise"].quantile(0.9)
        fall_cutoff = price_moves["prob_fall"].quantile(0.9)

    def render_one(row, card):
        replacement = squad_alert_checks.suggest_replacement(row, squad_ids, bank, all_players, team_limit_counts)
        st.markdown(f"**{card['position']} {row['web_name']}** — {card['basis']}: {card['detail']}")
        if replacement is None:
            st.caption(f"No budget-feasible same-position replacement found within your bank "
                       f"(£{bank/10:.1f}m). Manual look needed.")
            return

        st.markdown(f"→ Suggested: **{replacement['web_name']}** (£{replacement['now_cost']/10:.1f}m, "
                    f"{replacement['team_name']})")

        if not fixture_summary.empty:
            attacking = row["position"] in ("MID", "FWD")
            metric = "avg_attacking_difficulty" if attacking else "avg_defensive_difficulty"
            label = "attacking" if attacking else "defensive"
            out_name, in_name = team_id_to_name.get(row["team"]), replacement["team_name"]
            if out_name in fixture_summary.index and in_name in fixture_summary.index:
                fc1, fc2 = st.columns(2)
                fc1.metric(f"{row['web_name']} next 3 ({label})", f"{fixture_summary.loc[out_name, metric]:.2f}")
                fc2.metric(f"{replacement['web_name']} next 3 ({label})", f"{fixture_summary.loc[in_name, metric]:.2f}")

        if price_by_id is not None:
            if row["element"] in price_by_id.index:
                v = price_by_id.loc[row["element"], "prob_fall"]
                if pd.notna(v) and v >= fall_cutoff:
                    st.caption(f"💰 {row['web_name']}'s price looks likely to fall soon - "
                               f"consider selling before it does.")
            if replacement["id"] in price_by_id.index:
                v = price_by_id.loc[replacement["id"], "prob_rise"]
                if pd.notna(v) and v >= rise_cutoff:
                    st.caption(f"💰 {replacement['web_name']}'s price looks likely to rise soon - "
                               f"consider buying now before it does.")

    if red:
        st.error("🔴 CONFIRMED — official status or a released team sheet")
        for row, card in red:
            with st.container(border=True):
                render_one(row, card)
    if yellow:
        st.warning("🟡 EARLY WARNING — predicted only, could change before the deadline")
        for row, card in yellow:
            with st.container(border=True):
                render_one(row, card)


# ---------- My Team only: B3, the transfer planning engine ----------

def estimate_free_transfers(team_id: int) -> int:
    """FPL's public API has no direct 'free transfers currently available'
    field - every per-gameweek 'current' entry only carries event_transfers
    (transfers actually MADE that week), so this replays the accumulation
    rule (1/week, capped at multi_week_planner.MAX_FREE_TRANSFERS, chip
    weeks don't consume the bank) against that history.

    COULD NOT get a live populated example of get_entry_history()'s
    'current'/'chips' lists this session to confirm field names beyond
    'event_transfers' (already trusted elsewhere in this file, from the
    identical entry_history object team_picks_dataframe() returns) - checked
    several real Team IDs live on 2026-08-21 and 'current' was empty for
    every one of them, because the query window fell in the pre-season gap:
    the 2025-26 season had already rolled into 'past' as a season summary,
    and 2026-27's GW1 (deadline the same day) hadn't happened yet. Defaults
    to 1 whenever history is missing/unreadable, and the UI exposes this as
    an editable number, not an asserted fact - see render_transfer_advice()."""
    try:
        history_data = fpl_api.get_entry_history(team_id)
        current = (history_data or {}).get("current", [])
        if not current:
            return 1
        chip_events = {c.get("event") for c in (history_data.get("chips") or [])
                       if c.get("name") in ("wildcard", "freehit")}
        ft = 1
        for gw_entry in current:
            made = gw_entry.get("event_transfers", 0) or 0
            if gw_entry.get("event") not in chip_events:
                ft = max(ft - made, 0)
            ft = min(ft + 1, multi_week_planner.MAX_FREE_TRANSFERS)
        return ft
    except Exception:
        return 1


def estimate_chips_used(team_id: int, chip_windows: dict) -> set:
    """Same live-verification gap as estimate_free_transfers() above (see
    its docstring) - best-effort match of each logged chip's name/event
    against whichever chip_windows instance's window contains that event.
    Defaults to 'nothing used yet' on any failure - the safe direction,
    since the worst case is the plan suggesting a chip you've already
    burned, which you'd simply not follow, not a crash."""
    try:
        history_data = fpl_api.get_entry_history(team_id)
        used = set()
        for c in (history_data or {}).get("chips", []):
            name, event = c.get("name"), c.get("event")
            if name is None or event is None:
                continue
            for cid, window in chip_windows.items():
                if window["name"] == name and window["start_event"] <= event <= window["stop_event"]:
                    used.add(cid)
        return used
    except Exception:
        return set()


CANDIDATE_POOL_TOP_N_PER_POSITION = 40  # generous multiple of squad_optimizer's own quotas
                                          # (GK:2/DEF:5/MID:5/FWD:3) - plenty of real transfer-target
                                          # diversity while sharply cutting the ILP's variable count.
                                          # An unfiltered solve (all ~567 players) measured ~3.5min
                                          # locally, with unknown headroom on Streamlit Community
                                          # Cloud's free-tier CPU/RAM - not verified against the
                                          # actual deployed app, so this plus SOLVE_TIME_LIMIT_SECONDS
                                          # below are both real mitigations, not just UI disclosure.
SOLVE_TIME_LIMIT_SECONDS = 90  # bounded safety net ON TOP OF the pool filter above, not instead of
                                # it - CBC returns its best feasible solution so far if this is hit
                                # without having proven optimality; render_transfer_advice() discloses
                                # that rather than presenting a truncated solve with full confidence.
TRANSFER_PLAN_HORIZON_WEEKS = 3  # was 5 - each extra gameweek roughly doubles the ILP's decision
                                  # variables (squad/starting/captain/transfer per player PER week),
                                  # and the 5-week solve was regularly hitting SOLVE_TIME_LIMIT_SECONDS
                                  # without proving optimality (see conversation 2026-08-27: "5 weeks
                                  # is causing too much delay"). Only week 1 is ever actually commit-
                                  # ted to anyway - the rest is disclosed as a live forecast, rerun
                                  # weekly - so a shorter horizon trades a bit of lookahead (mainly
                                  # chip-timing foresight) for solves that reliably finish.


def filter_candidate_pool(preds_all: pd.DataFrame, current_squad_ids: set,
                           top_n: int = CANDIDATE_POOL_TOP_N_PER_POSITION) -> pd.DataFrame:
    """Top-N predicted_points per position, PLUS the user's own current squad
    regardless of rank - the squad must stay fully selectable no matter where
    its players land in the ranking, or the solver could be forced into
    recommending a sell it was never actually allowed to keep."""
    keep_ids = set(current_squad_ids)
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_pool = preds_all[preds_all["position"] == pos].nlargest(top_n, "predicted_points")
        keep_ids |= set(pos_pool["id"])
    return preds_all[preds_all["id"].isin(keep_ids)]


HIT_MARGIN_PCT = 0.06  # -4 is a big bite out of a typical week's score (e.g. ~8% of a 50-point
                        # week already) - requiring the projected edge to clear break-even by
                        # only a hair means a hit gets taken on noise as often as on real signal
                        # (squad_optimizer.py's backtested MAE is ~1pt/player/gameweek, so a
                        # 1-2pt "edge" is within the model's own known error bar). Padding the
                        # required edge by a further ~6% of a typical gameweek score - grounded
                        # in FPL's own live average_entry_score, not a made-up constant - pushes
                        # the bar to a comfortably real edge before spending the real, certain -4.


def typical_gameweek_score(_bootstrap: dict, default: float = 50.0) -> float:
    """FPL's own average_entry_score from the most recently finished gameweek - the live
    "what does a normal week actually score" benchmark used to size HIT_MARGIN_PCT's points
    buffer. Falls back to `default` pre-season, before any gameweek has an average yet."""
    finished = [e for e in _bootstrap["events"] if e.get("finished") and e.get("average_entry_score")]
    if not finished:
        return default
    return float(max(finished, key=lambda e: e["id"])["average_entry_score"])


@st.cache_data(ttl=1800)
def solve_transfer_plan(squad_ids_tuple: tuple, bank: int, free_transfers: int, target_event: int,
                         team_id: int, _bootstrap: dict, _all_players: pd.DataFrame,
                         allow_hits: bool = True) -> pd.DataFrame:
    """Cached on the inputs that actually change the answer (squad, bank,
    free transfers, target gameweek, team id) - an MILP solve isn't free,
    so repeated views/clicks within the 30min window don't re-solve.
    Leading-underscore params: same "don't hash this, already stable"
    convention as the other cached loaders above."""
    base = load_base_predictions()
    preds_all = refresh_predictions_with_live_data(base, _bootstrap)
    squad_ids = set(squad_ids_tuple)
    preds_all = filter_candidate_pool(preds_all, squad_ids)

    bundle = load_match_model_bundle()
    avg_lambda = load_league_avg_lambda(bundle, tuple(t["name"] for t in _bootstrap["teams"]))
    fixtures_df = load_fixtures_df(_bootstrap)

    gameweeks = [g for g in range(target_event, target_event + TRANSFER_PLAN_HORIZON_WEEKS) if g <= 38]
    projection_table = multi_week_projections.build_projection_table(
        preds_all, fixtures_df, gameweeks, bundle, avg_lambda,
    )
    team_of = dict(zip(_all_players["id"], _all_players["team_name"]))
    chip_windows = multi_week_planner.chip_windows_from_bootstrap(_bootstrap["chips"])
    chips_already_used = estimate_chips_used(team_id, chip_windows)

    hit_confidence_margin = typical_gameweek_score(_bootstrap) * HIT_MARGIN_PCT
    return multi_week_planner.plan(
        projection_table, gameweeks, squad_ids, bank, free_transfers,
        chip_windows, chips_already_used, team_of,
        time_limit=SOLVE_TIME_LIMIT_SECONDS, allow_hits=allow_hits,
        hit_confidence_margin=hit_confidence_margin,
    )


def render_transfer_advice(key_prefix: str, team_id: int, picks_df: pd.DataFrame, history: dict, target_event: int,
                            is_ai: bool = False):
    """is_ai=True (the AI Team tab): this isn't advice FOR someone, the
    model IS this team's manager - copy speaks in those terms (a transfer
    decision it's making, not a recommendation it's handing you) rather
    than reusing My Team's "here's what you should do" framing."""
    if is_ai:
        st.markdown("##### This week's transfer decision")
        st.caption(f"What the model is bringing in this week, and why. Solves the next "
                   f"{TRANSFER_PLAN_HORIZON_WEEKS} gameweeks but only commits to this week's move - "
                   "the rest is a live forecast, rerun weekly (see the note below once it runs).")
    else:
        st.markdown("##### Transfer advice")
        st.caption(f"Who to bring in this week, and why. Solves the next {TRANSFER_PLAN_HORIZON_WEEKS} "
                   "gameweeks but only commits to this week's move - the rest is a live forecast, "
                   "rerun weekly (see the note below once it runs).")

    default_ft = estimate_free_transfers(team_id)
    c1, c2 = st.columns([2, 1])
    free_transfers = c1.number_input(
        "Free transfers the AI has this week" if is_ai else "Free transfers available",
        min_value=0, max_value=multi_week_planner.MAX_FREE_TRANSFERS,
        value=default_ft, key=f"{key_prefix}_ft",
        help="FPL's API doesn't directly expose this - this is a best-effort estimate from the "
             "account's transfer history. Correct it if it's wrong; the plan below solves around "
             "whatever you enter.",
    )
    typical_score = typical_gameweek_score(bootstrap)
    hit_bar = multi_week_planner.HIT_COST + typical_score * HIT_MARGIN_PCT
    if is_ai:
        # No human risk preference to defer to here - the model runs this team, so whether a
        # hit is worth it is left to its own cost/benefit math each week (the objective already
        # requires clearing hit_bar, not just break-even), rather than a fixed human-set policy.
        # My Team gets an explicit opt-in below instead, since that's Ryan's own personal risk
        # call, not the model's.
        allow_hits = True
        st.caption(f"Point hits aren't fixed on or off here - a real -4 is ~{4/typical_score:.0%} of "
                   f"a typical week (~{typical_score:.0f} pts, FPL's own average this season), so the "
                   f"model only takes one when its projected gain over the horizon clears a real "
                   f"edge (~{hit_bar:.1f} pts), not a razor-thin one.")
    else:
        allow_hits = st.checkbox(
            "Allow point hits (-4 pts per transfer beyond your free ones)",
            value=False, key=f"{key_prefix}_allow_hits",
            help=f"Off by default: the plan below only ever uses transfers you actually have "
                 "banked (or an unlimited amount during a wildcard/free hit week). Turn this on "
                 "to let the solver take a hit - but only when its projected gain over the "
                 f"horizon clears a real edge (~{hit_bar:.1f} pts), not just the literal -4 "
                 f"break-even - a real hit is too big a chunk of a typical week (~{typical_score:.0f} "
                 "pts) to spend on a razor-thin modelled edge.",
        )
    if c2.button("Decide this week's move" if is_ai else "Get transfer advice", key=f"{key_prefix}_plan_btn"):
        st.session_state[f"{key_prefix}_show_plan"] = True

    if not st.session_state.get(f"{key_prefix}_show_plan"):
        st.caption("Solves an ILP over a filtered candidate pool (top predicted-points players per "
                   f"position, plus the squad{'' if is_ai else ' you loaded'}) across "
                   f"{TRANSFER_PLAN_HORIZON_WEEKS} gameweeks, bounded to {SOLVE_TIME_LIMIT_SECONDS}s "
                   "- not a quick lookup. Cached for 30 minutes afterward, so repeated views don't "
                   "re-solve.")
        return

    with st.spinner(f"Solving the next {TRANSFER_PLAN_HORIZON_WEEKS} gameweeks "
                    f"(bounded to {SOLVE_TIME_LIMIT_SECONDS}s)..."):
        try:
            plan_df = solve_transfer_plan(
                tuple(sorted(picks_df["element"])), int(history.get("bank", 0)), int(free_transfers),
                target_event, team_id, bootstrap, all_players, allow_hits,
            )
        except Exception as e:
            st.error(f"Couldn't build a transfer plan right now: {e}")
            return

    if not plan_df.attrs.get("proven_optimal", True):
        st.warning(f"The solver hit its {SOLVE_TIME_LIMIT_SECONDS}s time limit before proving this "
                   "is the best possible plan - it's the best one found in time, not a converged "
                   "solution. Treat it as a strong candidate, not a certainty.")

    this_week = plan_df.iloc[0]
    if this_week["transfers_in"]:
        st.markdown(f"**IN:** {', '.join(this_week['transfers_in'])}")
        st.markdown(f"**OUT:** {', '.join(this_week['transfers_out'])}")
    else:
        st.markdown("No transfer recommended this week.")
    if this_week["hits_taken"]:
        st.caption(f"Costs **-{this_week['hits_taken'] * multi_week_planner.HIT_COST} points** in hits - "
                   f"the plan judged the gain worth it.")
    if this_week["captain"]:
        st.markdown(f"**Captain:** {this_week['captain']}")
    if pd.notna(this_week["chip"]):
        st.markdown(f"**Chip this week:** {this_week['chip']}")

    future_chip = next((r for _, r in plan_df.iloc[1:].iterrows() if pd.notna(r["chip"])), None)
    if future_chip is not None:
        st.caption(f"Plan currently looks toward using **{future_chip['chip']}** around "
                   f"GW{future_chip['gameweek']}.")

    st.caption("Known limitation: this reuses one static season-baseline prediction scaled by "
               "fixture difficulty for the whole horizon, rather than re-estimating player quality "
               "week by week - real and disclosed, not hidden (see NEXT_STEPS.md).")


# ---------- My Team only: B4, best XI from the existing 15 ----------

def render_best_xi_recommendation(picks_df: pd.DataFrame):
    st.markdown("##### Best XI this week")
    st.caption("Who should start from your existing 15 - not a transfer suggestion, see Transfer "
               "advice above for that.")

    base = load_base_predictions()
    preds_all = refresh_predictions_with_live_data(base, bootstrap)
    squad_ids = set(picks_df["element"])
    squad_for_opt = preds_all[preds_all["id"].isin(squad_ids)]
    if len(squad_for_opt) < 15:
        st.caption("Not all 15 squad players have a model prediction available (e.g. a very recent "
                   "signing) - best-XI comparison skipped this week.")
        return

    xi_rec, bench_rec, formation_rec, captain_rec, vice_rec = pick_starting_xi(squad_for_opt)
    actual_starting_ids = set(picks_df[picks_df["multiplier"] > 0]["element"])
    rec_ids = set(xi_rec["id"])

    if rec_ids == actual_starting_ids:
        st.success(f"Your starting XI is already optimal this week ({formation_rec}).")
        return

    actual_predicted = squad_for_opt[squad_for_opt["id"].isin(actual_starting_ids)]["predicted_points"].sum()
    gain = xi_rec["predicted_points"].sum() - actual_predicted
    name_of = dict(zip(squad_for_opt["id"], squad_for_opt["web_name"]))
    bench_in = [name_of.get(i, "?") for i in (rec_ids - actual_starting_ids)]
    start_out = [name_of.get(i, "?") for i in (actual_starting_ids - rec_ids)]
    st.info(f"Your best XI this week is a **{formation_rec}** — start **{', '.join(bench_in)}** "
            f"instead of **{', '.join(start_out)}** for **+{gain:.1f}** predicted points.")


# ---------- My Team only: B5, season performance history ----------

def render_season_history(team_id: int):
    st.markdown("##### Season so far")

    history_data = fpl_api.get_entry_history(team_id)
    current = (history_data or {}).get("current", [])
    if not current:
        st.caption("No gameweek history yet this season.")
        return

    # Field names per fpl_api.get_entry_history()'s own docstring ("full
    # season history... plus chip usage log") - could NOT get a live
    # populated example of the 'current' list this session (see
    # estimate_free_transfers()'s docstring for why - every real Team ID
    # checked on 2026-08-21 had an empty list, pre-season). Building
    # defensively: only render whichever well-documented per-gameweek field
    # is actually present, skip the rest rather than guess.
    df = pd.DataFrame(current)
    if "event" in df.columns and "points" in df.columns:
        st.line_chart(df.set_index("event")[["points"]].rename(columns={"points": "Points"}))
    if "event" in df.columns and "overall_rank" in df.columns:
        st.line_chart(df.set_index("event")[["overall_rank"]].rename(columns={"overall_rank": "Overall rank"}))
        st.caption("Lower is better on the rank chart - it isn't inverted, read it that way.")
    if "event" not in df.columns:
        st.caption("Season history came back in an unexpected shape this run - skipping the chart "
                   "rather than guess at field names.")


# ---------- My Team only: B7, differentials watchlist (stretch) ----------

def render_watchlist():
    st.markdown("##### Differentials to watch")
    st.caption("Discovery only, not tied to your squad - players the model rates higher than their "
               "ownership suggests.")

    base = load_base_predictions()
    preds_all = refresh_predictions_with_live_data(base, bootstrap) if live_ok else base
    pool = preds_all[preds_all["available"]]
    analyzed = risk_analyzer.analyze(pool, target_rank_direction="chasing")
    top = risk_analyzer.top_picks_by_position(analyzed, n=3)

    show = top[["web_name", "position", "now_cost", "predicted_points", "ownership_pct", "tier"]].copy()
    show["now_cost"] = (show["now_cost"] / 10).map("£{:.1f}m".format)
    show["predicted_points"] = show["predicted_points"].round(1)
    st.dataframe(show, hide_index=True, use_container_width=True)


with tab_ai:
    st.subheader("The AI-run team")
    ai_team_id = get_ai_team_id()

    if ai_team_id is None:
        st.caption("Every decision here is made by the model. No human input. "
                   "**Not yet submitted to a real FPL account** — this is the current recommendation.")

        base = load_base_predictions()
        preds = refresh_predictions_with_live_data(base, bootstrap) if live_ok else base

        squad = pick_squad(preds, budget=BUDGET)
        xi, bench, formation, captain, vice = pick_starting_xi(squad)

        spend = squad["now_cost"].sum() / 10
        c1, c2, c3 = st.columns(3)
        c1.metric("Squad value", f"£{spend:.1f}m / £100.0m")
        c2.metric("Formation", formation)
        c3.metric("Predicted XI points", f"{xi['predicted_points'].sum():.1f}")

        if live_ok:
            render_squad_pitch(xi, bench, "id", "predicted_points", bootstrap, all_players,
                                gw["id"], captain_id=captain["id"], vice_id=vice["id"])
        else:
            st.info("Live FPL API unreachable - starting-likelihood flags need it, so the pitch "
                    "view is skipped this run. Try again shortly.")

    else:
        st.caption("🔒 Live and locked — this reads directly from the AI's real FPL account. "
                   "Nobody viewing this page, including friends, can change it. Changes only "
                   "happen when Ryan executes an alert email on the real account.")

        if not live_ok:
            st.error("Can't reach the live FPL API right now, so the AI team can't be displayed.")
        else:
            entry = fpl_api.get_entry(ai_team_id)
            if entry is None:
                st.error("AI_TEAM_ID is set but no matching FPL team was found — check the ID.")
            else:
                st.success(f"**{entry['name']}**")
                current_event = gw["id"] if gw["is_current"] else max(gw["id"] - 1, 1)
                result = fpl_api.team_picks_dataframe(ai_team_id, current_event, bootstrap)
                if result is None:
                    st.info(f"No squad locked in yet for Gameweek {current_event}.")
                else:
                    picks_df, history = result
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total points", entry.get("summary_overall_points") or "N/A yet")
                    c2.metric("Overall rank", format_rank(entry))
                    c3.metric("Bank", f"£{history['bank'] / 10:.1f}m")
                    c4.metric("Team value", f"£{history['value'] / 10:.1f}m")

                    xi = picks_df[picks_df["multiplier"] > 0]
                    bench = picks_df[picks_df["multiplier"] == 0]
                    captain_row = picks_df[picks_df["is_captain"]]
                    vice_row = picks_df[picks_df["is_vice_captain"]]
                    render_squad_pitch(
                        xi, bench, "element", "ep_next", bootstrap, all_players, current_event,
                        captain_id=captain_row["element"].iloc[0] if not captain_row.empty else None,
                        vice_id=vice_row["element"].iloc[0] if not vice_row.empty else None,
                    )

                    render_transfer_advice("ai", ai_team_id, picks_df, history, current_event, is_ai=True)


with tab_mine:
    st.subheader("Your team")

    if not st.user.is_logged_in:
        st.info("Log in with Google to save your Team ID once and have it remembered on "
                 "every future visit - no re-entering it, no browser-local tricks that "
                 "break if you switch devices.")
        st.button("Log in with Google", on_click=st.login, key="mine_login")
        st.caption("Or skip login and just look your team up for this visit:")
        render_team_lookup("mine_guest", show_suggested_changes=True)
    else:
        top_l, top_r = st.columns([4, 1])
        top_l.caption(f"Logged in as {st.user.email}")
        if top_r.button("Log out", key="mine_logout"):
            st.logout()

        try:
            profile = manager_store.get_manager_profile(st.user.sub)
            store_error = None
        except Exception as e:
            profile, store_error = None, e

        if store_error is not None:
            st.error(f"Couldn't reach the persistent store right now: {store_error}")
            st.caption("Falling back to one-off entry - it just won't be remembered this time.")
            render_team_lookup("mine", show_suggested_changes=True)
        else:
            saved_team_id = profile.get("fpl_team_id") if profile else None

            if saved_team_id is None:
                st.markdown("**First time here** — enter your FPL Team ID once and it'll be "
                            "remembered for next time.")
                new_id = st.number_input(
                    "Your FPL Team ID", min_value=1, step=1, key="mine_first_id",
                    help="Find this in the FPL site URL when viewing 'Pick Team' or 'Gameweek history'.",
                )
                if st.button("Save and load my team", key="mine_save_first"):
                    manager_store.save_fpl_team_id(st.user.sub, st.user.email, st.user.name, int(new_id))
                    st.rerun()
            else:
                st.success(f"Team ID **{saved_team_id}** remembered — no need to re-enter it.")

                def _on_toggle_email_alerts():
                    manager_store.save_email_alerts_enabled(st.user.sub, st.session_state.mine_email_alerts)

                st.checkbox(
                    "Email me if my starting XI has an injury/suspension issue",
                    value=profile.get("email_alerts_enabled", False),
                    key="mine_email_alerts",
                    on_change=_on_toggle_email_alerts,
                    help="Opt-in only - nobody gets emailed without checking this themselves, "
                         "and unchecking it stops future alerts immediately.",
                )
                with st.expander("Change my saved Team ID"):
                    updated_id = st.number_input(
                        "New FPL Team ID", min_value=1, step=1, value=saved_team_id, key="mine_update_id",
                    )
                    if st.button("Update saved Team ID", key="mine_save_update"):
                        manager_store.save_fpl_team_id(st.user.sub, st.user.email, st.user.name, int(updated_id))
                        st.rerun()
                render_team_lookup("mine", default_id=saved_team_id, show_suggested_changes=True)


with tab_friends:
    st.subheader("Friends")
    st.caption("Anyone can enter their FPL Team ID here — no login needed, this is public FPL data.")

    # Logged in: source the list from Supabase (manager_store.py) so it
    # survives a refresh/new device - the previous session_state-only
    # behavior is now only the logged-out fallback. Not logged in: unchanged
    # from before, session-only.
    persisted = False
    if st.user.is_logged_in:
        try:
            friend_ids = manager_store.get_friend_team_ids(st.user.sub)
            persisted = True
        except Exception as e:
            st.error(f"Couldn't reach the persistent store right now: {e}")
            st.caption("Falling back to session-only for this visit.")
            friend_ids = st.session_state.setdefault("friend_ids", [])
    else:
        friend_ids = st.session_state.setdefault("friend_ids", [])

    new_id = st.number_input("Add a friend's Team ID", min_value=1, step=1, key="new_friend_id")
    if st.button("Add friend"):
        if int(new_id) not in friend_ids:
            if persisted:
                try:
                    manager_store.add_friend_team_id(st.user.sub, int(new_id))
                except Exception as e:
                    st.error(f"Couldn't save that friend: {e}")
                else:
                    st.rerun()
            else:
                st.session_state.friend_ids.append(int(new_id))
                st.rerun()

    if not friend_ids:
        st.info("No friends added yet.")
    else:
        friend_tabs = st.tabs([f"ID {fid}" for fid in friend_ids])
        for fid, ftab in zip(friend_ids, friend_tabs):
            with ftab:
                # Persisting the list means the old implicit reset (refresh
                # clears it) no longer applies - without a real delete path,
                # once added, a friend would be stuck forever.
                if st.button("Remove", key=f"remove_friend_{fid}"):
                    if persisted:
                        manager_store.remove_friend_team_id(st.user.sub, fid)
                    else:
                        st.session_state.friend_ids.remove(fid)
                    st.rerun()
                # render_team_lookup()'s own click-to-load gate is untouched -
                # persisting more friend IDs must not cause more automatic
                # FPL API calls on page load, only the pre-fill changes.
                render_team_lookup(f"friend_{fid}", default_id=fid)

    if persisted:
        st.caption("Your friends list is saved to your account and stays across visits.")
    else:
        st.caption("Log in on My Team to keep this list saved across visits - for now, it resets "
                   "when the page is refreshed.")
