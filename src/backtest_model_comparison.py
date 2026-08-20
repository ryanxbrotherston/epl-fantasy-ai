"""
backtest_model_comparison.py — does XGBoost beat the shipped
RandomForestRegressor on the exact same held-out season, same features,
same recency weighting? Real comparison, not a guess - see NEXT_STEPS.md's
data/model audit section for why this was asked and the answer either way.

Reuses train_points_model.py's own data-loading/feature-building functions
directly (not reimplemented) so this is a genuinely apples-to-apples swap:
identical train/test seasons, identical rolling features, identical
recency-weighted sample_weight - only the model class changes.

LightGBM was NOT tested here (XGBoost only) - a scope decision made for
time, not because LightGBM was ruled out for a real reason. Worth doing
if XGBoost's result here makes a second gradient-boosted comparison
seem worthwhile.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from feature_config import CORE_STATS, OPTIONAL_STATS, POSITIONS, ALL_HISTORICAL_SEASONS
from train_points_model import load_season, build_rolling_features, TRAIN_SEASONS, TEST_SEASON

warnings.filterwarnings("ignore")


def main():
    print(f"Loading {len(TRAIN_SEASONS)} training seasons + held-out {TEST_SEASON} "
          f"(identical to train_points_model.py)...")
    train_frames = [load_season(s) for s in TRAIN_SEASONS]
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = load_season(TEST_SEASON)

    train_feat = build_rolling_features(train_df)
    test_feat = build_rolling_features(test_df)

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

    season_order = {s: i for i, s in enumerate(ALL_HISTORICAL_SEASONS)}
    test_idx = season_order[TEST_SEASON]
    DECAY = 0.5
    sample_weight = train_feat["season"].map(lambda s: DECAY ** (test_idx - season_order[s])).values

    print("\nTraining RandomForestRegressor (shipped config, unchanged)...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

    print("Training XGBRegressor (default-ish, lightly regularized to match RF's intent)...")
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    xgb_mae = mean_absolute_error(y_test, xgb.predict(X_test))

    baseline_mae = mean_absolute_error(y_test, test_feat["roll_total_points"])

    print(f"\n=== Held-out {TEST_SEASON} MAE (points/gameweek, lower is better) ===")
    print(f"Naive rolling-average baseline: {baseline_mae:.4f}")
    print(f"RandomForestRegressor (shipped): {rf_mae:.4f}")
    print(f"XGBRegressor:                    {xgb_mae:.4f}")
    diff = rf_mae - xgb_mae
    print(f"\nXGBoost {'beats' if diff > 0 else 'loses to'} RandomForest by {abs(diff):.4f} pts/GW "
          f"({abs(diff)/rf_mae*100:.1f}% {'better' if diff > 0 else 'worse'}).")


if __name__ == "__main__":
    main()
