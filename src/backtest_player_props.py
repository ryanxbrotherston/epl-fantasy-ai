"""
backtest_player_props.py — honest validation of player_props.py against a
real held-out season, not just sanity-check spot-checks.

Uses 2025-26 (never trained on by either the points model or match model)
as the test season. For every player-fixture in that season with enough
prior rolling history to generate a prediction, compares the predicted
P(scores)/P(assists) against what actually happened.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from feature_config import ROLLING_WINDOW, CORE_STATS, OPTIONAL_STATS, POSITION_ID_MAP
from match_model import predict_lambda, load_fixtures, TEST_SEASON
from player_props import player_fixture_props, league_average_lambda

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"
MODEL_DIR = BASE / "models"


def build_test_season_rolling(season=TEST_SEASON) -> pd.DataFrame:
    """Same rolling-feature logic as training, applied within the test season
    itself so we get a roll_* value for every gameweek an established player
    has enough prior history for - this is what a live in-season engine would
    produce, just computed here from the historical CSV instead of the live API."""
    df = pd.read_csv(DATA_DIR / season / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    for col in CORE_STATS + OPTIONAL_STATS:
        if col not in df.columns:
            df[col] = 0
    df["position"] = df["position"].replace({"GKP": "GK"})
    df = df.sort_values(["element", "GW"]).reset_index(drop=True)

    stat_cols = CORE_STATS + OPTIONAL_STATS
    grouped = df.groupby("element")[stat_cols]
    shifted = grouped.shift(1)
    shifted["element"] = df["element"]
    rolled = shifted.groupby("element")[stat_cols].rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolled = rolled.droplevel(0)
    rolled.columns = [f"roll_{c}" for c in stat_cols]

    out = pd.concat([df[["element", "name", "team", "opponent_team", "was_home", "GW",
                          "goals_scored", "assists", "position"]], rolled], axis=1)
    return out.dropna(subset=["roll_minutes"])


def main():
    bundle = joblib.load(MODEL_DIR / "match_model.pkl")
    model, encoder = bundle["model"], bundle["encoder"]

    teams_df = pd.read_csv(DATA_DIR / TEST_SEASON / "teams.csv", encoding="utf-8-sig")[["id", "name"]]
    team_id_to_name = dict(zip(teams_df["id"], teams_df["name"]))
    avg_lambda = league_average_lambda(bundle, teams_df["name"].tolist())

    test_data = build_test_season_rolling()
    print(f"Evaluating {len(test_data)} player-gameweek rows from {TEST_SEASON}...")

    rows = []
    for _, r in test_data.iterrows():
        team_name = r["team"]  # already a name string in this dataset - only opponent_team is a numeric id
        opp_name = team_id_to_name.get(r["opponent_team"])
        if team_name is None or opp_name is None:
            continue
        team_lambda = predict_lambda(model, encoder, team_name, opp_name, int(r["was_home"]))
        if np.isnan(team_lambda):
            continue

        props = player_fixture_props(r, team_lambda=team_lambda, league_avg_lambda=avg_lambda)
        rows.append({
            "p_scores": props["p_scores"], "actual_scored": r["goals_scored"] > 0,
            "p_assists": props["p_assists"], "actual_assisted": r["assists"] > 0,
        })

    results = pd.DataFrame(rows)
    n = len(results)

    scorer_brier = brier_score_loss(results["actual_scored"].astype(int), results["p_scores"])
    assist_brier = brier_score_loss(results["actual_assisted"].astype(int), results["p_assists"])

    naive_score_rate = results["actual_scored"].mean()
    naive_assist_rate = results["actual_assisted"].mean()
    naive_scorer_brier = brier_score_loss(results["actual_scored"].astype(int), [naive_score_rate] * n)
    naive_assist_brier = brier_score_loss(results["actual_assisted"].astype(int), [naive_assist_rate] * n)

    print(f"\nRows evaluated: {n}")
    print(f"\nAnytime scorer - Brier: {scorer_brier:.4f} (naive league-average-rate baseline: {naive_scorer_brier:.4f})")
    print(f"Anytime assist - Brier: {assist_brier:.4f} (naive league-average-rate baseline: {naive_assist_brier:.4f})")
    print(f"\nActual base rates: scored={naive_score_rate:.1%}, assisted={naive_assist_rate:.1%}")

    # calibration check: bucket predictions into quintiles, compare predicted vs actual rate per bucket
    results["score_bucket"] = pd.qcut(results["p_scores"], 5, duplicates="drop")
    calib = results.groupby("score_bucket", observed=True).agg(
        predicted=("p_scores", "mean"), actual=("actual_scored", "mean"), n=("actual_scored", "size")
    )
    print("\nScorer calibration by quintile (predicted vs actual rate - should track closely if well-calibrated):")
    print(calib.to_string())


if __name__ == "__main__":
    main()
