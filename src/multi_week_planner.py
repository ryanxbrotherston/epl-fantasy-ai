"""
multi_week_planner.py — rolling multi-gameweek transfer and chip optimizer.

Genuinely different from squad_optimizer.py's single-week ILP: decision
variables exist PER GAMEWEEK across the horizon, squad composition can only
change via explicit transfer variables from one week to the next, free
transfers accumulate (capped at 5 available in any given week, per FPL's
own game_settings.max_extra_free_transfers=4 + the 1 you get that week),
and each chip is only usable within its real eligibility window (pulled from
bootstrap-static's chips list, not hardcoded) and at most once per instance.

Discipline (see NEXT_STEPS.md): solve the WHOLE horizon every time this
runs, but only commit to the CURRENT gameweek's decision. Everything further
out is a live forecast, not a promise — rerun this weekly (or whenever
new information arrives) and let later weeks' plans change as double/blank
gameweeks get confirmed and form/injuries develop.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pulp

BASE = Path(__file__).resolve().parent.parent

SQUAD_SIZE = 15
POSITION_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
TEAM_LIMIT = 3
HIT_COST = 4  # points per transfer beyond available free transfers
SQUAD_QUALITY_BONUS = 0.02  # tiny per-player-per-gameweek nudge toward carrying available (not
                             # injured/suspended) players in the 15-man SQUAD, not just the
                             # starting 11. Rationale: a benched injured player already scores 0
                             # either way (starting[]<=squad[] means bench never hits the points
                             # objective), so on pure points there's no pressure to ever clear one
                             # out - but a "solid overall team" carries real bench depth/insurance,
                             # not permanent passengers (see conversation 2026-08-27). Deliberately
                             # tiny (0.02 * up to 3 gameweeks = 0.06 total) so it only ever breaks
                             # a genuine tie or nudges a free, no-hit-needed transfer - it must
                             # never be big enough to justify a hit or override a real points edge
                             # (hit_confidence_margin alone is ~7pts; this is two orders smaller).
MAX_FREE_TRANSFERS = 5  # 1/week + up to 4 banked (FPL's max_extra_free_transfers=4)


def chip_windows_from_bootstrap(bootstrap_chips: list[dict]) -> dict:
    """{chip_instance_id: {'name': 'wildcard', 'start_event': 2, 'stop_event': 19}}
    Two instances of most chips exist (one per half-season) - each is a
    SEPARATE usable-once chip, not the same chip playable twice."""
    return {c["id"]: {"name": c["name"], "start_event": c["start_event"], "stop_event": c["stop_event"]}
            for c in bootstrap_chips}


def plan(
    projection_table: pd.DataFrame,  # from multi_week_projections.py: id, web_name, position, team_name, now_cost, gw{N} columns
    gameweeks: list[int],
    current_squad_ids: set,
    current_bank: int,  # tenths of a million
    current_free_transfers: int,
    chip_windows: dict,
    chips_already_used: set,  # chip instance ids already burned this season
    team_of: dict,  # {player_id: team_name} for club-limit constraints
    time_limit: float | None = None,  # seconds - a bounded safety net, not a substitute for
                                       # keeping the candidate pool small (see app.py's
                                       # solve_transfer_plan(), which filters projection_table
                                       # before this ever gets called) - CBC can still return its
                                       # best feasible solution so far rather than the proven
                                       # optimum if this is hit; see the sol_status note below.
    allow_hits: bool = True,  # if False, transfers are hard-capped at whatever's free
                               # each week (banked + wildcard/free hit) instead of letting
                               # the solver trade -4pts/transfer for predicted-points gain -
                               # some users would rather bank a mediocre gain than ever
                               # take a hit, regardless of what the model says it's worth.
    hit_confidence_margin: float = 0.0,  # extra points of projected edge required ON TOP OF
                               # HIT_COST before a hit is taken at all - see app.py's
                               # typical_gameweek_score() for how this is set. Rationale: -4 is
                               # a real cost, but "projected gain > 4" alone means the solver
                               # takes a hit on a razor-thin modelled edge, and the projection
                               # itself carries real error (squad_optimizer.py's backtested MAE
                               # is ~1pt/player/gameweek) - a hit should clear a comfortably
                               # larger bar than pure break-even before it's worth the real,
                               # certain -4. Only changes the DECISION threshold, not the
                               # reported cost - hits_taken is still billed at the true HIT_COST
                               # in _extract_plan() below and in app.py's display.
):
    prob = pulp.LpProblem("multi_week_fpl_plan", pulp.LpMaximize)
    players = projection_table["id"].tolist()
    gw_cols = {gw: f"gw{gw}" for gw in gameweeks}

    squad = {(p, gw): pulp.LpVariable(f"squad_{p}_{gw}", cat="Binary") for p in players for gw in gameweeks}
    starting = {(p, gw): pulp.LpVariable(f"start_{p}_{gw}", cat="Binary") for p in players for gw in gameweeks}
    captain = {(p, gw): pulp.LpVariable(f"cap_{p}_{gw}", cat="Binary") for p in players for gw in gameweeks}
    transfer_in = {(p, gw): pulp.LpVariable(f"tin_{p}_{gw}", cat="Binary") for p in players for gw in gameweeks}
    transfer_out = {(p, gw): pulp.LpVariable(f"tout_{p}_{gw}", cat="Binary") for p in players for gw in gameweeks}

    wildcard = {(cid, gw): pulp.LpVariable(f"wc_{cid}_{gw}", cat="Binary")
                for cid, c in chip_windows.items() if c["name"] == "wildcard" for gw in gameweeks}
    freehit = {(cid, gw): pulp.LpVariable(f"fh_{cid}_{gw}", cat="Binary")
               for cid, c in chip_windows.items() if c["name"] == "freehit" for gw in gameweeks}
    bboost = {(cid, gw): pulp.LpVariable(f"bb_{cid}_{gw}", cat="Binary")
              for cid, c in chip_windows.items() if c["name"] == "bboost" for gw in gameweeks}
    triple_capt = {(cid, gw): pulp.LpVariable(f"tc_{cid}_{gw}", cat="Binary")
                   for cid, c in chip_windows.items() if c["name"] == "3xc" for gw in gameweeks}

    free_transfers_avail = {gw: pulp.LpVariable(f"fts_{gw}", lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer")
                             for gw in gameweeks}
    hits = {gw: pulp.LpVariable(f"hits_{gw}", lowBound=0, cat="Integer") for gw in gameweeks}

    any_wildcard_or_freehit = {gw: pulp.lpSum(
        [wildcard.get((cid, gw), 0) for cid in chip_windows if chip_windows[cid]["name"] == "wildcard"]
        + [freehit.get((cid, gw), 0) for cid in chip_windows if chip_windows[cid]["name"] == "freehit"]
    ) for gw in gameweeks}
    triple_capt_active = {gw: pulp.lpSum(
        triple_capt.get((cid, gw), 0) for cid in chip_windows if chip_windows[cid]["name"] == "3xc"
    ) for gw in gameweeks}

    # captain[(p,gw)] * triple_capt_active[gw] is a product of two decision
    # variables - not allowed in a linear program. Standard fix: an auxiliary
    # binary that's linearly constrained to equal that AND, since both sides
    # are already 0/1 (at most one triple-captain chip per week is enforced
    # below, so triple_capt_active is effectively binary too).
    capt_and_tc = {(p, gw): pulp.LpVariable(f"capttc_{p}_{gw}", cat="Binary") for p in players for gw in gameweeks}

    available_of = {
        row["id"]: bool(row["available"]) if "available" in projection_table.columns else True
        for _, row in projection_table.iterrows()
    }

    objective_terms = []
    for gw in gameweeks:
        col = gw_cols[gw]
        for p in players:
            pts = float(projection_table.loc[projection_table["id"] == p, col].iloc[0])
            base = pts * starting[(p, gw)]
            cap_bonus = pts * captain[(p, gw)]
            tc_bonus = pts * capt_and_tc[(p, gw)]
            objective_terms.append(base + cap_bonus + tc_bonus)
            if available_of.get(p, True):
                objective_terms.append(SQUAD_QUALITY_BONUS * squad[(p, gw)])
        objective_terms.append(-(HIT_COST + hit_confidence_margin) * hits[gw])
    prob += pulp.lpSum(objective_terms)

    # AND-linearization: capt_and_tc = captain AND triple_capt_active
    for p in players:
        for gw in gameweeks:
            prob += capt_and_tc[(p, gw)] <= captain[(p, gw)]
            prob += capt_and_tc[(p, gw)] <= triple_capt_active[gw]
            prob += capt_and_tc[(p, gw)] >= captain[(p, gw)] + triple_capt_active[gw] - 1

    for gw in gameweeks:
        prob += pulp.lpSum(squad[(p, gw)] for p in players) == SQUAD_SIZE
        prob += pulp.lpSum(starting[(p, gw)] for p in players) == 11
        prob += pulp.lpSum(captain[(p, gw)] for p in players) == 1

        for pos, quota in POSITION_QUOTAS.items():
            pos_players = [p for p in players if projection_table.loc[projection_table["id"] == p, "position"].iloc[0] == pos]
            prob += pulp.lpSum(squad[(p, gw)] for p in pos_players) == quota

        for team in set(team_of.values()):
            team_players = [p for p in players if team_of.get(p) == team]
            prob += pulp.lpSum(squad[(p, gw)] for p in team_players) <= TEAM_LIMIT

        for p in players:
            prob += starting[(p, gw)] <= squad[(p, gw)]
            prob += captain[(p, gw)] <= starting[(p, gw)]

        total_cost = pulp.lpSum(
            squad[(p, gw)] * float(projection_table.loc[projection_table["id"] == p, "now_cost"].iloc[0])
            for p in players
        )
        prob += total_cost <= current_bank + sum(
            float(projection_table.loc[projection_table["id"] == p, "now_cost"].iloc[0])
            for p in current_squad_ids if p in players
        )

    for i, gw in enumerate(gameweeks):
        for p in players:
            prev = squad[(p, gameweeks[i - 1])] if i > 0 else (1 if p in current_squad_ids else 0)
            prob += squad[(p, gw)] >= prev + transfer_in[(p, gw)] - transfer_out[(p, gw)] - any_wildcard_or_freehit[gw]
            prob += squad[(p, gw)] <= prev + transfer_in[(p, gw)] - transfer_out[(p, gw)] + any_wildcard_or_freehit[gw]
            prob += transfer_out[(p, gw)] <= prev
            # a player can't be sold and rebought in the same week - without this,
            # the solver can pick meaningless self-cancelling pairs whenever the
            # free-transfer budget has slack (net zero effect, but a confusing plan)
            prob += transfer_in[(p, gw)] + transfer_out[(p, gw)] <= 1

    for i, gw in enumerate(gameweeks):
        transfers_made = pulp.lpSum(transfer_in[(p, gw)] for p in players)
        if i == 0:
            prob += free_transfers_avail[gw] == current_free_transfers
        else:
            prev_gw = gameweeks[i - 1]
            prev_transfers_made = pulp.lpSum(transfer_in[(p, prev_gw)] for p in players)
            prob += free_transfers_avail[gw] >= free_transfers_avail[prev_gw] - prev_transfers_made + 1 - MAX_FREE_TRANSFERS * any_wildcard_or_freehit[prev_gw]
            prob += free_transfers_avail[gw] <= MAX_FREE_TRANSFERS

        if allow_hits:
            prob += hits[gw] >= transfers_made - free_transfers_avail[gw] - MAX_FREE_TRANSFERS * any_wildcard_or_freehit[gw]
        else:
            prob += hits[gw] == 0
            prob += transfers_made <= free_transfers_avail[gw] + MAX_FREE_TRANSFERS * any_wildcard_or_freehit[gw]

    for cid, c in chip_windows.items():
        var_dict = {"wildcard": wildcard, "freehit": freehit, "bboost": bboost, "3xc": triple_capt}[c["name"]]
        eligible_gws = [gw for gw in gameweeks if c["start_event"] <= gw <= c["stop_event"]]
        ineligible_gws = [gw for gw in gameweeks if gw not in eligible_gws]
        for gw in ineligible_gws:
            if (cid, gw) in var_dict:
                prob += var_dict[(cid, gw)] == 0
        if cid in chips_already_used:
            for gw in gameweeks:
                if (cid, gw) in var_dict:
                    prob += var_dict[(cid, gw)] == 0
        else:
            prob += pulp.lpSum(var_dict.get((cid, gw), 0) for gw in gameweeks) <= 1

    for gw in gameweeks:
        prob += any_wildcard_or_freehit[gw] <= 1
        prob += pulp.lpSum(
            [bboost.get((cid, gw), 0) for cid in chip_windows if chip_windows[cid]["name"] == "bboost"]
            + [triple_capt.get((cid, gw), 0) for cid in chip_windows if chip_windows[cid]["name"] == "3xc"]
        ) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    # CBC/PuLP quirk, confirmed empirically (not assumed from docs): when a
    # timeLimit cuts the solve short but CBC still has a feasible incumbent,
    # prob.status stays "Optimal" - that field alone can't distinguish a
    # proven-optimal solve from a truncated-but-usable one. prob.sol_status
    # is the real signal (pulp/apis/coin_api.py's get_status(): CBC's own
    # solution-file header "Stopped on time - objective value X" maps to
    # LpSolutionIntegerFeasible, not LpSolutionOptimal). Only genuinely
    # unusable outcomes (infeasible, unbounded, or a time-out with no
    # feasible solution at all) raise; a truncated-but-feasible one is
    # still returned; app.py's caller discloses that to the user rather
    # than presenting it with the same confidence as a converged solve.
    if status not in ("Optimal",) or prob.sol_status not in (
        pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible,
    ):
        raise RuntimeError(f"Solver status: {status} - horizon/constraints may be infeasible")

    plan_df = _extract_plan(gameweeks, players, projection_table, squad, starting, captain,
                             transfer_in, transfer_out, wildcard, freehit, bboost, triple_capt,
                             hits, free_transfers_avail, current_squad_ids)
    plan_df.attrs["proven_optimal"] = prob.sol_status == pulp.LpSolutionOptimal
    return plan_df


def _extract_plan(gameweeks, players, projection_table, squad, starting, captain,
                   transfer_in, transfer_out, wildcard, freehit, bboost, triple_capt,
                   hits, free_transfers_avail, current_squad_ids):
    plan_rows = []
    prev_squad_ids = set(current_squad_ids)
    for gw in gameweeks:
        # Derived from actual squad membership, NOT the transfer_in/transfer_out indicator
        # variables - during a wildcard/free hit week, any_wildcard_or_freehit's slack term
        # lets squad[] change freely without the solver having any incentive to also set those
        # indicators to match (they're not what the objective/constraints actually gate that
        # week), so a real squad-out-with-no-corresponding-transfer_out=1 is possible and DID
        # happen live: "IN: Thiago, Dubravka" with an empty "OUT:" during a wildcard week (see
        # conversation 2026-08-27). Comparing this week's squad to last week's squad directly
        # is correct regardless of which chip (if any) enabled the change.
        this_squad_ids = {p for p in players if squad[(p, gw)].value() == 1}
        ins = sorted(this_squad_ids - prev_squad_ids)
        outs = sorted(prev_squad_ids - this_squad_ids)
        prev_squad_ids = this_squad_ids
        cap = next((p for p in players if captain[(p, gw)].value() == 1), None)
        chip_used = None
        for cid, gw2 in wildcard:
            if gw2 == gw and wildcard[(cid, gw)].value() == 1:
                chip_used = "wildcard"
        for cid, gw2 in freehit:
            if gw2 == gw and freehit[(cid, gw)].value() == 1:
                chip_used = "free hit"
        for cid, gw2 in bboost:
            if gw2 == gw and bboost[(cid, gw)].value() == 1:
                chip_used = (chip_used + " + bench boost") if chip_used else "bench boost"
        for cid, gw2 in triple_capt:
            if gw2 == gw and triple_capt[(cid, gw)].value() == 1:
                chip_used = (chip_used + " + triple captain") if chip_used else "triple captain"

        name_of = lambda p: projection_table.loc[projection_table["id"] == p, "web_name"].iloc[0]
        plan_rows.append({
            "gameweek": gw,
            "transfers_in": [name_of(p) for p in ins],
            "transfers_out": [name_of(p) for p in outs],
            "transfers_in_ids": ins,   # same players as transfers_in, as ids - callers that need
            "transfers_out_ids": outs,  # to apply the swap unambiguously (e.g. feeding the
                                          # resulting squad into a best-XI pick) shouldn't have to
                                          # re-match on web_name, which isn't guaranteed unique.
            "hits_taken": int(hits[gw].value()),
            "captain": name_of(cap) if cap else None,
            "chip": chip_used,
            "free_transfers_available": int(free_transfers_avail[gw].value()),
        })
    return pd.DataFrame(plan_rows)
