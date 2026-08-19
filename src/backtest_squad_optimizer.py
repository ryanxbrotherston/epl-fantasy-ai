"""
backtest_squad_optimizer.py — does the price-bias correction in
squad_optimizer.py's PRICE_BIAS_CORRECTION actually pick better squads, on
real held-out data?

BACKGROUND - what was tried first and why it's not what shipped:
The README/NEXT_STEPS framed this as a "ceiling/variance" problem - the
optimizer maximizes average predicted points, which supposedly undervalues
explosive premiums. The first thing tried here was exactly that: a
variance-reward term built on player_props.py's Poisson framework
(predicted_points + risk_weight*points_std, so a high-scoring-rate player's
fat-tailed distribution nudges them up the rankings). Backtested honestly
against 2025-26 with the same fresh-pick-each-gameweek method this script
uses, it made things WORSE (avg actual XI+captain points/GW: 60.6 at
risk_weight=0, falling monotonically to 56.9 at risk_weight=1.0, 0/32
gameweeks improved). This makes sense in hindsight: picking a fixed-size (15
players, fixed position quotas) squad under a linear points payoff is an
expected-value-maximization problem, not a portfolio-variance problem -
rewarding variance just trades away real expected points for no offsetting
benefit, since nothing here has a convex/rank-based payoff for a single
gameweek's raw points.

So: what data actually IS true? Checking predicted_points against ACTUAL
points by price tier on 2025-26 revealed the real issue is a MEAN
calibration bias, not a variance one - the model+blend systematically
UNDER-predicts expensive players specifically:
  price tier   mean bias (actual - predicted), full 2025-26 season
  budget        +0.13 pts/GW
  mid           +0.56 pts/GW
  premium       +1.19 pts/GW
  elite         +1.35-1.6 pts/GW
This is very likely RandomForestRegressor's usual behavior of pulling
extreme predictions toward the bulk of the training distribution
(min_samples_leaf=5, max_depth=10 both regularize hard), diluted further by
blending in ep_next/ppg that don't fully correct it. PRICE_BIAS_CORRECTION
in squad_optimizer.py is a linear fit of that bias against price, which
DOES change squad selection (unlike a uniform rescale of predicted_points,
which - since squad size and position quotas are fixed cardinalities -
provably can't change which players an ILP like this one picks at all).

This script validates that correction out-of-sample: fits it on GW6-20,
evaluates fresh-pick-each-week squad selection (real pick_squad() /
pick_starting_xi() from squad_optimizer.py) on GW21-37, which the fit never
saw. Compares actual XI+captain points captured (captain doubled, matching
real FPL scoring) against the uncorrected baseline.

Known simplifications: no injury/availability filtering (not in the
historical per-gameweek file), no auto-subs for a 0-minute XI player, and
this is a "if you picked completely fresh from a full £100.0m budget every
single week" comparison, not season-long squad evolution with transfers -
it isolates the objective function's effect, which is exactly what changed.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_config import ROLLING_WINDOW, CORE_STATS, OPTIONAL_STATS, POSITIONS
from squad_optimizer import BLEND_WEIGHTS, pick_squad, pick_starting_xi, BUDGET

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"
MODEL_DIR = BASE / "models"
TEST_SEASON = "2025-26"

FIT_GAMEWEEKS = range(6, 21)   # correction is fit on this range only
EVAL_GAMEWEEKS = range(21, 38)  # ...and evaluated on this range, which the fit never saw


def build_test_season_rolling() -> pd.DataFrame:
    """Rolling features as of each gameweek within the test season itself,
    same logic as train_points_model.build_rolling_features - each row's
    roll_* columns only use gameweeks STRICTLY BEFORE that row's GW (shift(1)
    before the rolling mean), so this is honest pre-gameweek information."""
    df = pd.read_csv(DATA_DIR / TEST_SEASON / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    for col in CORE_STATS + OPTIONAL_STATS:
        if col not in df.columns:
            df[col] = 0
    df["position"] = df["position"].replace({"GKP": "GK"})

    # merged_gw.csv has two kinds of (element, GW) duplicates that both need
    # handling BEFORE the rolling window is built, or a double-gameweek/
    # duplicate-row player silently distorts what "trailing 5 gameweeks"
    # means for every later row of theirs: (1) exact duplicate rows - a real
    # data-file glitch, a handful of players have the identical row twice for
    # the same fixture; (2) genuine double gameweeks - two DIFFERENT fixture
    # rows for the same player in the same GW, which should sum, not
    # duplicate. drop_duplicates() strips (1); the groupby-sum below handles
    # (2). Flagged in NEXT_STEPS.md - the same unhandled duplication exists
    # in train_points_model.py/backtest_player_props.py's rolling-feature
    # builders, out of scope to fix (and retrain) here tonight.
    df = df.drop_duplicates()
    sum_cols = CORE_STATS + OPTIONAL_STATS + ["xP"]
    first_cols = [c for c in df.columns if c not in sum_cols + ["element", "GW"]]
    df = df.groupby(["element", "GW"], as_index=False).agg(
        {**{c: "sum" for c in sum_cols}, **{c: "first" for c in first_cols}}
    )
    df = df.sort_values(["element", "GW"]).reset_index(drop=True)

    stat_cols = CORE_STATS + OPTIONAL_STATS
    shifted = df.groupby("element")[stat_cols].shift(1)
    shifted["element"] = df["element"]
    rolled = shifted.groupby("element")[stat_cols].rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolled = rolled.droplevel(0)
    rolled.columns = [f"roll_{c}" for c in stat_cols]

    # ppg proxy: running average of ACTUAL total_points strictly before this GW
    running_ppg = df.groupby("element")["total_points"].apply(lambda s: s.shift(1).expanding().mean())
    running_ppg = running_ppg.reset_index(level=0, drop=True)

    out = pd.concat([
        df[["element", "name", "team", "GW", "position", "value", "total_points", "xP"]],
        rolled,
    ], axis=1)
    out["ppg_proxy"] = running_ppg
    return out.dropna(subset=["roll_minutes"])  # needs at least 1 prior appearance


def add_model_points(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_DIR / "points_model.pkl")
    with open(MODEL_DIR / "feature_columns.json") as f:
        meta = json.load(f)

    feat = pd.get_dummies(df, columns=["position"], prefix="pos")
    for p in POSITIONS:
        col = f"pos_{p}"
        if col not in feat.columns:
            feat[col] = 0
    feat["games_played_so_far"] = 38  # closing-form-style proxy, same convention as build_gw1_features.py

    df = df.copy()
    df["model_points"] = model.predict(feat[meta["features"]])
    return df


def build_predicted_points(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces squad_optimizer.load_predictions()'s blend, but with
    per-gameweek, fixture-specific inputs instead of the season-snapshot
    ones: FPL's own historical 'xP' column stands in for ep_next (it IS
    FPL's real expected-points estimate for that exact fixture), and a
    running points-per-game stands in for the season snapshot's ppg."""
    df = df.copy()
    ep_next = pd.to_numeric(df["xP"], errors="coerce").fillna(df["model_points"])
    ppg = pd.to_numeric(df["ppg_proxy"], errors="coerce").fillna(df["model_points"])
    df["predicted_points"] = (
        BLEND_WEIGHTS["model"] * df["model_points"]
        + BLEND_WEIGHTS["ep_next"] * ep_next
        + BLEND_WEIGHTS["ppg"] * ppg
    )
    return df


def score_gameweek(pool: pd.DataFrame, actuals: pd.Series) -> float | None:
    """Actual XI points (captain doubled) for the squad pick_squad()/
    pick_starting_xi() would have picked from this pool, using ONLY
    pre-gameweek information."""
    try:
        squad = pick_squad(pool, budget=BUDGET)
    except RuntimeError:
        return None  # infeasible (shouldn't happen with a full 20-team pool, but don't crash the sweep)
    xi, bench, formation, captain, vice = pick_starting_xi(squad)
    xi_actual = actuals.reindex(xi["id"]).fillna(0).sum()
    captain_actual = actuals.get(captain["id"], 0)
    return xi_actual + captain_actual  # captain's points counted a second time


def main():
    print("Building rolling features across the 2025-26 season...")
    rolling = build_test_season_rolling()
    rolling = add_model_points(rolling)
    rolling = build_predicted_points(rolling)
    rolling["id"] = rolling["element"]
    rolling["now_cost"] = rolling["value"]
    rolling["available"] = True

    fit = rolling[rolling["GW"].isin(FIT_GAMEWEEKS)].copy()
    fit["bias"] = fit["total_points"] - fit["predicted_points"]
    b, a = np.polyfit(fit["value"].values, fit["bias"].values, 1)
    print(f"\nCorrection fit on GW{FIT_GAMEWEEKS.start}-{FIT_GAMEWEEKS.stop - 1}: "
          f"bias = {a:.4f} + {b:.6f} * price  (n={len(fit)})")

    raw_scores, corrected_scores, gws_tested = [], [], []
    for gw in EVAL_GAMEWEEKS:
        pool = rolling[rolling["GW"] == gw].reset_index(drop=True)
        if len(pool) < 100 or pool["team"].nunique() < 10:
            continue  # not enough of the league present this GW to fill a real squad
        gws_tested.append(gw)
        actuals = pool.set_index("id")["total_points"]

        raw_scores.append(score_gameweek(pool, actuals))

        corrected = pool.copy()
        corrected["predicted_points"] = corrected["predicted_points"] + (a + b * corrected["value"])
        corrected_scores.append(score_gameweek(corrected, actuals))

    print(f"\nEvaluated on GW{gws_tested[0]}-{gws_tested[-1]} ({len(gws_tested)} gameweeks, "
          f"never used to fit the correction above)\n")
    print(f"{'':>20} | {'avg XI+captain pts/GW':>22} | {'total':>8}")
    print(f"{'uncorrected baseline':>20} | {np.mean(raw_scores):>22.2f} | {np.sum(raw_scores):>8.1f}")
    print(f"{'price-corrected':>20} | {np.mean(corrected_scores):>22.2f} | {np.sum(corrected_scores):>8.1f}")

    diffs = [c - r for r, c in zip(raw_scores, corrected_scores)]
    print(f"\nMean per-GW lift: {np.mean(diffs):+.3f} pts, GWs improved: "
          f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}")


if __name__ == "__main__":
    main()
