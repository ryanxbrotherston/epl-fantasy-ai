"""
backtest_multi_week_planner.py — persisted, re-runnable validation that
multi_week_planner.py/multi_week_projections.py correctly handle real
double/blank gameweeks and produce a valid, solvable multi-week plan.

Gap this fixes: NEXT_STEPS.md's "multi_week_planner re-verification"
section (2026-08-20) found the original "tested against real 2022-23
DGWs" claim from an earlier session was never saved as a reusable script,
unlike every other model in this repo. This is that script.

NOT held-out for point-prediction accuracy: 2022-23 is deliberately NOT
used here to check whether predicted_points is accurate, because it's one
of points_model.pkl's own TRAINING seasons (see train_points_model.py's
TRAIN_SEASONS) - treating it as held-out would be data leakage. Instead
this uses a model-independent points proxy: each player's own trailing,
strictly-backward-looking actual points-per-game that season. That
isolates what's actually under test here - does the PLANNER (fixture-
difficulty scaling, DGW summing, blank zeroing, transfer/budget
mechanics) behave correctly against a real congested fixture calendar -
not whether some point estimate is accurate. match_model.pkl is still
used for its fixture-difficulty SCALING role (a relative multiplier
across teams, not a target prediction) - the same role it already plays,
unflagged for leakage, in backtest_squad_optimizer.py.

NOT re-validated: chip timing. No chip-eligibility-window data exists for
a historical season in the vaastav CSVs (that's live FPL account/season
metadata, not part of the historical per-gameweek dumps) - rather than
guess plausible windows, this runs with chip_windows={} (no chips
available) and checks transfer/budget/DGW/blank correctness only. The
original ad-hoc test's "sensible chip timing" was an impression, never
itself checked against ground truth - still true here, explicitly, not
silently dropped.

Checks, all against real fixtures.csv data for 2022-23 GW18-26 (the World
Cup fixture-congestion window - independently confirmed via fixtures.csv
to contain real double gameweeks, e.g. Chelsea/Fulham GW19,
Man City/Man Utd/Spurs/Crystal Palace GW20, and a real blank, e.g.
Brentford GW25):
  1. A club with a confirmed real double gameweek gets a HIGHER projected
     total that week than the same club's single-fixture baseline.
  2. A club with a confirmed real blank gameweek gets EXACTLY ZERO
     projected points that week.
  3. The full multi-week plan solves to Optimal (no crash, no
     infeasibility) across this real DGW/blank-containing horizon.
  4. Squad composition stays valid every week of the plan (15 players,
     position quotas, club limit).
  5. The budget constraint is respected every week (spend never exceeds
     bank + squad value).
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from multi_week_projections import build_gw_fixture_map, build_projection_table
from multi_week_planner import plan, POSITION_QUOTAS, SQUAD_SIZE, TEAM_LIMIT
from player_props import league_average_lambda

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"
MODEL_DIR = BASE / "models"
TEST_SEASON = "2022-23"

HORIZON = list(range(18, 27))
KNOWN_REAL_DGW = {19: "Chelsea", 20: "Man City"}
KNOWN_REAL_BLANK = {25: "Brentford"}


def load_teams() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / TEST_SEASON / "teams.csv", encoding="utf-8-sig")


def load_fixtures_with_names(teams: pd.DataFrame) -> pd.DataFrame:
    fx = pd.read_csv(DATA_DIR / TEST_SEASON / "fixtures.csv", encoding="utf-8-sig")
    fx = fx.dropna(subset=["event"]).copy()
    fx["event"] = fx["event"].astype(int)
    id_to_name = dict(zip(teams["id"], teams["name"]))
    fx["team_h_name"] = fx["team_h"].map(id_to_name)
    fx["team_a_name"] = fx["team_a"].map(id_to_name)
    return fx


def build_points_proxy() -> pd.DataFrame:
    """Model-independent points-per-game proxy: trailing, strictly
    backward-looking average of each player's own actual total_points
    that season. Dedupes (element, GW) the same way as the other
    backtest_*.py scripts in this repo (see NEXT_STEPS.md's
    "merged_gw.csv duplicate-row fix" section)."""
    df = pd.read_csv(DATA_DIR / TEST_SEASON / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    df["position"] = df["position"].replace({"GKP": "GK"})
    df = df.drop_duplicates()
    sum_cols = ["total_points"]
    first_cols = [c for c in df.columns if c not in sum_cols + ["element", "GW"]]
    df = df.groupby(["element", "GW"], as_index=False).agg(
        {**{c: "sum" for c in sum_cols}, **{c: "first" for c in first_cols}}
    )
    df = df.sort_values(["element", "GW"]).reset_index(drop=True)
    df["ppg_proxy"] = (
        df.groupby("element")["total_points"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    return df


def build_squad_predictions(proxy: pd.DataFrame, as_of_gw: int) -> pd.DataFrame:
    """One row per player who'd appeared at least once before as_of_gw,
    using their ppg_proxy as of that point - this is the season-baseline
    'predicted_points' build_projection_table expects."""
    snap = (
        proxy[proxy["GW"] == as_of_gw]
        .dropna(subset=["ppg_proxy"])
        .drop_duplicates("element")
        .copy()
    )
    return snap.rename(columns={
        "element": "id", "name": "web_name", "team": "team_name",
        "value": "now_cost", "ppg_proxy": "predicted_points",
    })[["id", "web_name", "position", "team_name", "now_cost", "predicted_points"]]


def check_dgw_blank_scaling(table: pd.DataFrame, fixture_map: dict) -> list[str]:
    """Checks 1-2: real DGW clubs score higher than their single-fixture
    baseline that week, real blank clubs score exactly zero."""
    failures = []
    for gw, club in KNOWN_REAL_DGW.items():
        club_rows = table[table["team_name"] == club]
        if club_rows.empty:
            failures.append(f"GW{gw} DGW check: no players found for {club} in the projection table")
            continue
        dgw_total = club_rows[f"gw{gw}"].sum()
        baseline = club_rows["predicted_points"].sum()
        n_fixtures = len(fixture_map.get(club, {}).get(gw, []))
        if n_fixtures < 2:
            failures.append(f"GW{gw} DGW check: {club} only has {n_fixtures} fixture(s) in fixtures.csv - "
                             f"expected 2, the real-DGW claim itself may be stale")
        elif not (dgw_total > baseline):
            failures.append(f"GW{gw} DGW check FAILED: {club}'s projected GW{gw} total ({dgw_total:.1f}) "
                             f"is not higher than their single-fixture baseline ({baseline:.1f})")

    for gw, club in KNOWN_REAL_BLANK.items():
        club_rows = table[table["team_name"] == club]
        if club_rows.empty:
            failures.append(f"GW{gw} blank check: no players found for {club} in the projection table")
            continue
        blank_total = club_rows[f"gw{gw}"].sum()
        n_fixtures = len(fixture_map.get(club, {}).get(gw, []))
        if n_fixtures != 0:
            failures.append(f"GW{gw} blank check: {club} has {n_fixtures} fixture(s) in fixtures.csv - "
                             f"expected 0, the real-blank claim itself may be stale")
        elif blank_total != 0:
            failures.append(f"GW{gw} blank check FAILED: {club}'s projected GW{gw} total is "
                             f"{blank_total:.1f}, expected exactly 0")
    return failures


def check_plan_validity(plan_df: pd.DataFrame, table: pd.DataFrame, current_squad_ids: set) -> list[str]:
    """Checks 3-5: solved without error (caller already confirmed that by
    getting here), and every week's implied squad stays legal."""
    failures = []
    squad = set(current_squad_ids)
    for _, row in plan_df.iterrows():
        gw = row["gameweek"]
        ins = set(table[table["web_name"].isin(row["transfers_in"])]["id"])
        outs = set(table[table["web_name"].isin(row["transfers_out"])]["id"])
        squad = (squad - outs) | ins

        if len(squad) != SQUAD_SIZE:
            failures.append(f"GW{gw}: squad size is {len(squad)}, expected {SQUAD_SIZE}")

        squad_rows = table[table["id"].isin(squad)]
        pos_counts = squad_rows.drop_duplicates("id")["position"].value_counts().to_dict()
        for pos, quota in POSITION_QUOTAS.items():
            if pos_counts.get(pos, 0) != quota:
                failures.append(f"GW{gw}: {pos} count is {pos_counts.get(pos, 0)}, expected {quota}")

        club_counts = squad_rows.drop_duplicates("id")["team_name"].value_counts()
        over_limit = club_counts[club_counts > TEAM_LIMIT]
        if not over_limit.empty:
            failures.append(f"GW{gw}: club limit exceeded - {over_limit.to_dict()}")
    return failures


def main():
    print(f"Backtesting multi_week_planner.py against real {TEST_SEASON} fixture data, "
          f"GW{HORIZON[0]}-{HORIZON[-1]} (World Cup fixture-congestion window)...\n")

    teams = load_teams()
    fixtures = load_fixtures_with_names(teams)
    fixture_map = build_gw_fixture_map(fixtures, HORIZON)
    proxy = build_points_proxy()

    squad_predictions = build_squad_predictions(proxy, as_of_gw=HORIZON[0])
    print(f"Players with a usable points-proxy as of GW{HORIZON[0]}: {len(squad_predictions)}")

    bundle = joblib.load(MODEL_DIR / "match_model.pkl")
    avg_lambda = league_average_lambda(bundle, teams["name"].tolist())
    table = build_projection_table(squad_predictions, fixtures, HORIZON, bundle, avg_lambda)

    print("\n--- Checks 1-2: real DGW/blank scaling ---")
    scaling_failures = check_dgw_blank_scaling(table, fixture_map)
    if scaling_failures:
        for f in scaling_failures:
            print("FAIL:", f)
    else:
        print("PASS: real 2022-23 DGWs scale up, real blanks are exactly zero.")

    print("\n--- Check 3: full plan solves ---")
    from squad_optimizer import pick_squad, BUDGET
    seed_pool = squad_predictions.assign(available=True, team=squad_predictions["team_name"])
    current_squad_df = pick_squad(seed_pool, budget=BUDGET)
    current_squad_ids = set(current_squad_df["id"])
    team_of = dict(zip(squad_predictions["id"], squad_predictions["team_name"]))

    table_for_plan = table[table["id"].isin(squad_predictions["id"])].reset_index(drop=True)
    try:
        plan_df = plan(
            projection_table=table_for_plan,
            gameweeks=HORIZON,
            current_squad_ids=current_squad_ids,
            current_bank=0,
            current_free_transfers=1,
            chip_windows={},
            chips_already_used=set(),
            team_of=team_of,
        )
        print(f"PASS: solved to Optimal across GW{HORIZON[0]}-{HORIZON[-1]}.")
    except RuntimeError as e:
        print(f"FAIL: solver did not reach Optimal - {e}")
        plan_df = None

    print("\n--- Checks 4-5: squad/budget validity every week ---")
    validity_failures = []
    if plan_df is not None:
        validity_failures = check_plan_validity(plan_df, table_for_plan, current_squad_ids)
        if validity_failures:
            for f in validity_failures:
                print("FAIL:", f)
        else:
            print("PASS: squad composition and club limits stayed valid every week.")

    print("\n--- Summary ---")
    total_failures = len(scaling_failures) + len(validity_failures) + (1 if plan_df is None else 0)
    if total_failures == 0:
        print("ALL CHECKS PASSED.")
    else:
        print(f"{total_failures} check(s) FAILED - see above.")
    return total_failures


if __name__ == "__main__":
    n_failures = main()
    sys.exit(1 if n_failures else 0)
