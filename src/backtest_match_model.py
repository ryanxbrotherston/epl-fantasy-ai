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


def main():
    model, encoder = train_and_save()
    test_fixtures = load_fixtures(TEST_SEASON)
    print(f"\nBacktesting on {len(test_fixtures)} matches from {TEST_SEASON}...\n")

    rows = []
    for _, fx in test_fixtures.iterrows():
        lh = predict_lambda(model, encoder, fx["home_name"], fx["away_name"], 1)
        la = predict_lambda(model, encoder, fx["away_name"], fx["home_name"], 0)
        if np.isnan(lh) or np.isnan(la):
            continue  # promoted team with no prior-season history

        probs = fixture_probabilities(lh, la)
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

    results = pd.DataFrame(rows)
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

    # naive baselines for comparison: always predict the league's actual home-win rate as p_home_win, etc.
    naive_home_rate = (results["actual_result"] == "H").mean()
    naive_draw_rate = (results["actual_result"] == "D").mean()
    naive_away_rate = (results["actual_result"] == "A").mean()
    naive_brier = np.mean([
        brier_score_loss((results["actual_result"] == cls).astype(int), [rate] * n)
        for cls, rate in [("H", naive_home_rate), ("D", naive_draw_rate), ("A", naive_away_rate)]
    ])

    print(f"Matches evaluated: {n} (some early-season promoted-team fixtures skipped - no prior history)")
    print(f"\nMatch winner accuracy: {accuracy:.1%}")
    print(f"Match winner Brier score: {brier:.4f}  (naive 'always predict league-average rate' baseline: {naive_brier:.4f})")
    print(f"Clean sheet Brier score - home: {cs_brier_home:.4f}, away: {cs_brier_away:.4f}")
    print(f"Total goals MAE: {goals_mae:.3f} goals/match")
    print(f"\nActual result distribution: H={naive_home_rate:.1%} D={naive_draw_rate:.1%} A={naive_away_rate:.1%}")


if __name__ == "__main__":
    main()
