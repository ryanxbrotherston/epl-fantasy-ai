"""
price_predictor.py — forecasts whether a player's price is likely to
rise/fall, from net-transfer momentum relative to their ownership.

IMPORTANT SCOPE NOTE: FPL's actual price-change algorithm runs on DAILY net
transfers and isn't publicly documented. The historical data this is trained
and backtested on is WEEKLY resolution only (one snapshot per gameweek) -
that's what's available, not a design choice. This means:
  - What's built and validated here: "will this player's price likely be
    higher/lower after this gameweek", from that week's aggregate transfer
    momentum. Genuinely useful for weekly transfer planning.
  - What this does NOT do: tell you a price is about to move TONIGHT, which
    is what people actually want for "get them in before the rise". That
    needs live daily polling accumulated through the season (the hourly
    GitHub Action can start collecting this once live - see NEXT_STEPS.md).
    Don't oversell this as more than it is.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"
MODEL_DIR = BASE / "models"

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"


def load_price_history(season: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / season / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    df = df.sort_values(["element", "GW"])
    df["prev_value"] = df.groupby("element")["value"].shift(1)
    df["price_change"] = np.sign(df["value"] - df["prev_value"])
    df["transfer_momentum"] = df["transfers_balance"] / df["selected"].replace(0, np.nan)
    df = df.dropna(subset=["prev_value", "transfer_momentum"])
    df["season"] = season
    return df


def build_and_backtest():
    train_df = pd.concat([load_price_history(s) for s in TRAIN_SEASONS], ignore_index=True)
    test_df = load_price_history(TEST_SEASON)

    X_train = train_df[["transfer_momentum"]]
    X_test = test_df[["transfer_momentum"]]

    results = {}
    for label, direction in [("rise", 1), ("fall", -1)]:
        y_train = (train_df["price_change"] == direction).astype(int)
        y_test = (test_df["price_change"] == direction).astype(int)

        # NOT class_weight="balanced" - that inflates predicted probabilities
        # to compensate for the rare-event imbalance, which wrecks Brier/
        # calibration even though the model's ranking ability (AUC) is fine.
        # Natural class weights keep the probabilities honestly calibrated;
        # AUC (threshold-independent) is the fairer ranking metric here given
        # how rare a rise/fall actually is in any given week.
        model = LogisticRegression()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        # top-decile precision: of the players the model ranks as MOST likely
        # to move, what fraction actually did - this is the metric that
        # matches how this would actually be used (a "watch these players"
        # ticker, not a single fixed-threshold yes/no call)
        top_decile_cutoff = np.quantile(probs, 0.9)
        top_decile_actual_rate = y_test[probs >= top_decile_cutoff].mean()

        brier = brier_score_loss(y_test, probs)
        naive_brier = brier_score_loss(y_test, [y_train.mean()] * len(y_test))
        auc = roc_auc_score(y_test, probs)

        results[label] = {"model": model, "brier": brier, "naive_brier": naive_brier, "auc": auc,
                           "top_decile_actual_rate": top_decile_actual_rate,
                           "actual_rate": y_test.mean()}

    import joblib
    joblib.dump({k: v["model"] for k, v in results.items()}, MODEL_DIR / "price_model.pkl")
    return results


if __name__ == "__main__":
    results = build_and_backtest()
    for label, r in results.items():
        print(f"\n=== Price {label} ===")
        print(f"Brier: {r['brier']:.4f} (naive baseline: {r['naive_brier']:.4f})")
        print(f"AUC-ROC: {r['auc']:.3f} (0.5 = no better than random, 1.0 = perfect ranking)")
        print(f"Top-decile precision: {r['top_decile_actual_rate']:.1%} actual rate among the top 10% "
              f"most-flagged players (vs {r['actual_rate']:.1%} base rate = "
              f"{r['top_decile_actual_rate']/r['actual_rate']:.1f}x lift)")
        print(f"Actual base rate this season: {r['actual_rate']:.1%}")
    print(f"\nSaved to models/price_model.pkl")
