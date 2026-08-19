"""
backtest_match_model.py — honest validation of match_model.py against a
season it never trained on. Same discipline as the points model: chronological
holdout, real metrics, no peeking.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).parent))
from match_model import (
    train_and_save, load_fixtures, predict_lambda, fixture_probabilities, TEST_SEASON,
)


def evaluate(model, encoder, test_fixtures, rho: float) -> pd.DataFrame:
    rows = []
    for _, fx in test_fixtures.iterrows():
        lh = predict_lambda(model, encoder, fx["home_name"], fx["away_name"], 1)
        la = predict_lambda(model, encoder, fx["away_name"], fx["home_name"], 0)
        if np.isnan(lh) or np.isnan(la):
            continue  # promoted team with no prior-season history

        probs = fixture_probabilities(lh, la, rho=rho)
        actual_result = "H" if fx["team_h_score"] > fx["team_a_score"] else (
            "A" if fx["team_a_score"] > fx["team_h_score"] else "D"
        )
        actual_home_cs = fx["team_a_score"] == 0
        actual_away_cs = fx["team_h_score"] == 0
        predicted_result = max(
            [("H", probs["p_home_win"]), ("D", probs["p_draw"]), ("A", probs["p_away_win"])],
            key=lambda t: t[1],
        )[0]

        rows.append({
            "actual_result": actual_result, "predicted_result": predicted_result,
            "p_home_win": probs["p_home_win"], "p_draw": probs["p_draw"], "p_away_win": probs["p_away_win"],
            "p_home_cs": probs["p_home_clean_sheet"], "actual_home_cs": actual_home_cs,
            "p_away_cs": probs["p_away_clean_sheet"], "actual_away_cs": actual_away_cs,
            "actual_total_goals": fx["team_h_score"] + fx["team_a_score"],
            "predicted_total_goals": lh + la,
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, label: str):
    n = len(results)
    accuracy = (results["actual_result"] == results["predicted_result"]).mean()

    # Brier score for the 3-way result (one-hot actual vs predicted probs), averaged across classes
    brier = np.mean([
        brier_score_loss((results["actual_result"] == cls).astype(int), results[f"p_{col}"])
        for cls, col in [("H", "home_win"), ("D", "draw"), ("A", "away_win")]
    ])

    cs_brier_home = brier_score_loss(results["actual_home_cs"].astype(int), results["p_home_cs"])
    cs_brier_away = brier_score_loss(results["actual_away_cs"].astype(int), results["p_away_cs"])
    goals_mae = (results["actual_total_goals"] - results["predicted_total_goals"]).abs().mean()

    # draw-specific Brier - the score Dixon-Coles' low-score correction is specifically meant to sharpen
    draw_brier = brier_score_loss((results["actual_result"] == "D").astype(int), results["p_draw"])

    print(f"\n--- {label} ---")
    print(f"Match winner accuracy: {accuracy:.1%}")
    print(f"Match winner Brier score: {brier:.4f}")
    print(f"Draw Brier score: {draw_brier:.4f}")
    print(f"Clean sheet Brier score - home: {cs_brier_home:.4f}, away: {cs_brier_away:.4f}")
    print(f"Total goals MAE: {goals_mae:.3f} goals/match")
    return {"n": n, "accuracy": accuracy, "brier": brier, "draw_brier": draw_brier}


def main():
    model, encoder, rho = train_and_save()
    test_fixtures = load_fixtures(TEST_SEASON)
    print(f"\nBacktesting on {len(test_fixtures)} matches from {TEST_SEASON}...")

    results_indep = evaluate(model, encoder, test_fixtures, rho=0.0)
    results_dc = evaluate(model, encoder, test_fixtures, rho=rho)
    n = len(results_indep)

    naive_home_rate = (results_indep["actual_result"] == "H").mean()
    naive_draw_rate = (results_indep["actual_result"] == "D").mean()
    naive_away_rate = (results_indep["actual_result"] == "A").mean()
    naive_brier = np.mean([
        brier_score_loss((results_indep["actual_result"] == cls).astype(int), [rate] * n)
        for cls, rate in [("H", naive_home_rate), ("D", naive_draw_rate), ("A", naive_away_rate)]
    ])

    print(f"Matches evaluated: {n} (some early-season promoted-team fixtures skipped - no prior history)")
    print(f"Naive 'always predict league-average rate' Brier baseline: {naive_brier:.4f}")
    print(f"Fitted Dixon-Coles rho: {rho:.4f}")

    summarize(results_indep, "Independent Poisson (rho=0, old behavior)")
    summarize(results_dc, f"Dixon-Coles corrected (rho={rho:.4f})")

    print(f"\nActual result distribution: H={naive_home_rate:.1%} D={naive_draw_rate:.1%} A={naive_away_rate:.1%}")


if __name__ == "__main__":
    main()
