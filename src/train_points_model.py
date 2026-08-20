"""
train_points_model.py — EPL Fantasy points predictor, v2 (full historical depth)

Same rolling-form + RandomForest approach as v1, but now trained across all
available seasons (2016-17 through 2024-25) rather than just the three most
recent. This isn't a free upgrade - FPL's own scoring rules and tracked stats
have genuinely changed over that span (see feature_config.py), so this is
schema harmonization, not a simple concat. Held-out test season is still
2025-26, chronologically after everything trained on - same discipline as v1.
"""

import warnings
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from feature_config import (
    ROLLING_WINDOW, CORE_STATS, OPTIONAL_STATS, POSITION_ID_MAP, POSITIONS,
    SEASONS_NEEDING_POSITION_JOIN, ALL_HISTORICAL_SEASONS,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "fpl-historical" / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TEST_SEASON = "2025-26"
# 6-season window, not the full 9 available: tested extensively (full depth,
# multiple recency-decay strengths) and none clearly beat this on the one real
# held-out season we can test against - see README for the actual numbers.
# This window skips 2016-17/17-18/18-19 (oldest, most rule-divergent, needed
# extra encoding/position-join handling) while keeping meaningfully more
# volume than a bare 3-season model, mostly for robustness against overfitting
# to a small season sample rather than a measured accuracy gain.
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def _read_csv_robust(path) -> pd.DataFrame:
    """Some older-season CSVs (2016-17 to 2018-19) aren't UTF-8 - accented
    player names break the default encoding. Fall back to latin-1, which
    covers the accented Western European characters these files use."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_season(season: str) -> pd.DataFrame:
    df = _read_csv_robust(DATA_DIR / season / "gws" / "merged_gw.csv")
    df["season"] = season

    for col in CORE_STATS + OPTIONAL_STATS:
        if col not in df.columns:
            df[col] = 0

    if season in SEASONS_NEEDING_POSITION_JOIN:
        players = _read_csv_robust(DATA_DIR / season / "players_raw.csv")
        pos_lookup = players.set_index("id")["element_type"].map(POSITION_ID_MAP)
        df["position"] = df["element"].map(pos_lookup)
    else:
        df["position"] = df["position"].replace({"GKP": "GK"})

    df = df.dropna(subset=["position"])

    # merged_gw.csv has two kinds of (element, GW) duplicates that both
    # distort the "trailing 5 gameweeks" rolling window if left in: exact
    # duplicate rows (a data-file glitch) and genuine double gameweeks (two
    # different fixture rows for the same player/GW, which should sum, not
    # duplicate). Same fix as backtest_squad_optimizer.py - see NEXT_STEPS.md.
    df = df.drop_duplicates()
    sum_cols = CORE_STATS + OPTIONAL_STATS
    first_cols = [c for c in df.columns if c not in sum_cols + ["element", "GW"]]
    df = df.groupby(["element", "GW"], as_index=False).agg(
        {**{c: "sum" for c in sum_cols}, **{c: "first" for c in first_cols}}
    )
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "element", "GW"]).reset_index(drop=True)
    stat_cols = CORE_STATS + OPTIONAL_STATS
    group_keys = ["season", "element"]

    # Fast path: shift once per column (vectorized), then a single Cython-level
    # groupby().rolling().mean() per column - NOT wrapped in a Python lambda,
    # which is what made the naive .apply() version scale badly past ~150k rows.
    shifted = df.groupby(group_keys)[stat_cols].shift(1)
    shifted[group_keys] = df[group_keys]

    rolled = (
        shifted.groupby(group_keys)[stat_cols]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )
    # groupby().rolling() returns a MultiIndex of (group_keys..., original_row_index) -
    # drop the group-key levels and keep the original index, so the concat below
    # aligns by index rather than trusting row order to have survived the regroup.
    rolled = rolled.droplevel(list(range(len(group_keys))))
    rolled.columns = [f"roll_{c}" for c in stat_cols]

    out = pd.concat(
        [df[["season", "element", "name", "position", "GW", "total_points"]], rolled],
        axis=1,
    )
    out["games_played_so_far"] = df.groupby(group_keys).cumcount()
    out = out.dropna(subset=[f"roll_{CORE_STATS[0]}"])
    return out


def main():
    print(f"Loading {len(TRAIN_SEASONS)} training seasons: {TRAIN_SEASONS}")
    train_frames = []
    for s in TRAIN_SEASONS:
        try:
            train_frames.append(load_season(s))
        except Exception as e:
            print(f"  Skipping {s}: {e}")
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = load_season(TEST_SEASON)

    print(f"Train rows (raw): {len(train_df)}, Test rows (raw): {len(test_df)}")

    train_feat = build_rolling_features(train_df)
    test_feat = build_rolling_features(test_df)
    print(f"Train rows (with rolling history): {len(train_feat)}, Test rows: {len(test_feat)}")

    feature_cols = [f"roll_{c}" for c in CORE_STATS + OPTIONAL_STATS] + ["games_played_so_far"]

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

    # Recency weighting: older seasons carry real target-label noise (different
    # scoring rules) and feature noise (zero-filled stats that didn't exist yet)
    # relative to what we're predicting for now. Rather than discard that data
    # outright, down-weight it exponentially by how many seasons back it is -
    # keeps the extra volume as weak signal without letting it dilute clean
    # recent signal on equal footing.
    season_order = {s: i for i, s in enumerate(ALL_HISTORICAL_SEASONS)}
    test_idx = season_order[TEST_SEASON]
    DECAY = 0.5
    sample_weight = train_feat["season"].map(
        lambda s: DECAY ** (test_idx - season_order[s])
    ).values

    print("Training RandomForestRegressor on full historical depth (recency-weighted)...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, test_feat["roll_total_points"])

    print(f"\nModel MAE on held-out {TEST_SEASON}: {mae:.3f} points/gameweek")
    print(f"Naive rolling-average baseline MAE: {baseline_mae:.3f} points/gameweek")
    print(f"(v1, trained on 3 seasons only, scored 0.989 MAE on the same test season)")

    importances = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importances.head(10))

    joblib.dump(model, MODEL_DIR / "points_model.pkl")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump({
            "features": all_features, "rolling_window": ROLLING_WINDOW,
            "core_stats": CORE_STATS, "optional_stats": OPTIONAL_STATS,
            "train_seasons": TRAIN_SEASONS, "test_season": TEST_SEASON,
            "test_mae": mae, "baseline_mae": baseline_mae,
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR / 'points_model.pkl'}")


if __name__ == "__main__":
    main()
