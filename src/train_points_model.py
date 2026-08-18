"""
train_points_model.py — EPL Fantasy points predictor

Same approach as the NRL player models: rolling-form features per player
over a trailing window of gameweeks, RandomForestRegressor trained on
multiple past seasons, chronological (not random) train/test split so we
never leak future gameweeks into training.

Data source: vaastav/Fantasy-Premier-League (community-maintained,
gameweek-by-gameweek FPL data, same role the scraped NRL Excel files played).

Output: models/points_model.pkl + models/feature_columns.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "fpl-historical" / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]  # train
TEST_SEASON = "2025-26"                             # held-out, chronologically after training seasons

ROLLING_WINDOW = 5  # trailing gameweeks - shorter than NRL's 10 since FPL "form" conventionally uses ~5 GW windows

CORE_STATS = [
    "minutes", "total_points", "ict_index", "threat", "creativity", "influence",
    "bps", "expected_goal_involvements", "expected_goals_conceded", "starts",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "bonus",
]
# only present from 2025-26 onward (new defensive contribution scoring rule)
OPTIONAL_STATS = ["defensive_contribution", "tackles", "clearances_blocks_interceptions", "recoveries"]

POSITIONS = ["GK", "DEF", "MID", "FWD"]
POSITION_MAP = {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / season / "gws" / "merged_gw.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["season"] = season
    for col in OPTIONAL_STATS:
        if col not in df.columns:
            df[col] = 0
    df["position"] = df["position"].map(POSITION_MAP).fillna(df["position"])
    # sort key: season order is chronological by construction of TRAIN_SEASONS/TEST_SEASON
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each player, compute trailing-window rolling means of core stats
    BEFORE the current gameweek (shift(1) so we never use the target GW's own
    stats to predict itself), then set target = this GW's total_points."""
    df = df.sort_values(["season", "element", "GW"]).reset_index(drop=True)
    stat_cols = CORE_STATS + OPTIONAL_STATS

    grouped = df.groupby(["season", "element"], group_keys=False)

    rolling_frames = []
    for col in stat_cols:
        roll = grouped[col].apply(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
        rolling_frames.append(roll.rename(f"roll_{col}"))

    rolling_df = pd.concat(rolling_frames, axis=1)
    out = pd.concat([df[["season", "element", "name", "position", "GW", "total_points", "value"]], rolling_df], axis=1)

    # games played so far this season (recency/reliability signal)
    out["games_played_so_far"] = grouped.cumcount()

    # drop rows with no history yet (first GW a player appears - nothing to roll from)
    out = out.dropna(subset=[f"roll_{CORE_STATS[0]}"])
    return out


def main():
    print("Loading seasons...")
    train_df = pd.concat([load_season(s) for s in TRAIN_SEASONS], ignore_index=True)
    test_df = load_season(TEST_SEASON)

    print(f"Train rows (raw): {len(train_df)}, Test rows (raw): {len(test_df)}")

    train_feat = build_rolling_features(train_df)
    test_feat = build_rolling_features(test_df)

    print(f"Train rows (with rolling history): {len(train_feat)}, Test rows: {len(test_feat)}")

    feature_cols = [f"roll_{c}" for c in CORE_STATS + OPTIONAL_STATS] + ["games_played_so_far"]

    # one-hot position
    train_feat = pd.get_dummies(train_feat, columns=["position"], prefix="pos")
    test_feat = pd.get_dummies(test_feat, columns=["position"], prefix="pos")
    pos_cols = [f"pos_{p}" for p in POSITIONS]
    for c in pos_cols:
        if c not in train_feat.columns:
            train_feat[c] = 0
        if c not in test_feat.columns:
            test_feat[c] = 0

    all_features = feature_cols + pos_cols

    X_train, y_train = train_feat[all_features], train_feat["total_points"]
    X_test, y_test = test_feat[all_features], test_feat["total_points"]

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, test_feat["roll_total_points"])  # naive "just use recent form" baseline

    print(f"Model MAE on held-out {TEST_SEASON}: {mae:.3f} points/gameweek")
    print(f"Naive rolling-average baseline MAE: {baseline_mae:.3f} points/gameweek")

    importances = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importances.head(10))

    import joblib
    joblib.dump(model, MODEL_DIR / "points_model.pkl")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump({"features": all_features, "rolling_window": ROLLING_WINDOW,
                   "core_stats": CORE_STATS, "optional_stats": OPTIONAL_STATS}, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR / 'points_model.pkl'}")


if __name__ == "__main__":
    main()
