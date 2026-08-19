"""
backtest_risk_analyzer_minutes_floor.py — does MIN_ROLL_MINUTES actually
make risk_analyzer.py's differential picks better, on real held-out data?

NEXT_STEPS.md flagged this as a known gap: "differential picks skew toward
noisy low-minutes players" because predicted_points for a barely-used player
is small-sample noise (the same failure mode player_props.py had before its
own shrinkage fix), so ranking among them isn't meaningful. This backtests
the fix rather than just asserting it: for each gameweek in 2025-26, builds
the player pool as of that gameweek (pre-gameweek information only), runs
risk_analyzer.analyze() in 'chasing' mode (the differential-seeking path)
with and without the minutes floor, and compares the ACTUAL total_points the
top_picks_by_position() recommendations earned that specific gameweek.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_squad_optimizer import (
    build_test_season_rolling, add_model_points, build_predicted_points, DATA_DIR, TEST_SEASON,
)
from risk_analyzer import analyze, top_picks_by_position

warnings.filterwarnings("ignore")

TEST_GAMEWEEKS = range(6, 38)
MINUTES_FLOOR_CANDIDATES = [0, 30, 60, 90]  # 0 reproduces the old (no-floor) behavior


def load_ownership() -> pd.DataFrame:
    """selected_by_percent isn't produced by build_test_season_rolling (it
    only carries what squad selection needs) - pull FPL's raw 'selected'
    ownership count per (element, GW) here. analyze() only ever uses this
    column via within-price-band percentile RANKS, so a raw count works
    exactly as well as a true percentage for ranking purposes."""
    df = pd.read_csv(DATA_DIR / TEST_SEASON / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    df = df.drop_duplicates()
    return df.groupby(["element", "GW"], as_index=False).agg({"selected": "max"})


def score_picks(picks: pd.DataFrame, actuals: pd.Series) -> float:
    return actuals.reindex(picks["id"]).fillna(0).mean()


def main():
    print("Building rolling features across the 2025-26 season...")
    rolling = build_test_season_rolling()
    rolling = add_model_points(rolling)
    rolling = build_predicted_points(rolling)
    rolling["id"] = rolling["element"]
    rolling["now_cost"] = rolling["value"]

    ownership = load_ownership()
    rolling = rolling.merge(ownership, on=["element", "GW"], how="left")
    rolling["selected_by_percent"] = rolling["selected"].fillna(0)

    results = {w: [] for w in MINUTES_FLOOR_CANDIDATES}
    gws_tested = []

    for gw in TEST_GAMEWEEKS:
        pool = rolling[rolling["GW"] == gw].reset_index(drop=True)
        if len(pool) < 100:
            continue
        actuals = pool.set_index("id")["total_points"]
        gws_tested.append(gw)

        for floor in MINUTES_FLOOR_CANDIDATES:
            ranked = analyze(pool, target_rank_direction="chasing", min_roll_minutes=floor)
            if ranked.empty:
                continue
            picks = top_picks_by_position(ranked, n=3)
            results[floor].append(score_picks(picks, actuals))

    print(f"\nGameweeks evaluated: {len(gws_tested)} ({gws_tested[0]}-{gws_tested[-1]})\n")
    print(f"{'min_roll_minutes':>18} | {'avg actual pts/pick':>20} | {'n picks/GW (avg)':>18}")
    for floor in MINUTES_FLOOR_CANDIDATES:
        scores = [s for s in results[floor] if s is not None]
        avg = np.mean(scores)
        print(f"{floor:>18} | {avg:>20.3f} | {'~12':>18}")

    baseline, floored = results[0], results[60]
    diffs = [f - b for b, f in zip(baseline, floored)]
    print(f"\nfloor=60 vs floor=0 (no filter): mean lift {np.mean(diffs):+.3f} pts/pick, "
          f"GWs improved: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")


if __name__ == "__main__":
    main()
