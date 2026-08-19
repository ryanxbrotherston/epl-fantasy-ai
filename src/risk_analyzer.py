"""
risk_analyzer.py — template vs differential pick analysis.

Two things every FPL manager actually weighs against each other:
  - TEMPLATE picks: high-ownership players. Safe - if they haul, so does
    everyone else's team, so you don't gain rank. If they blank, you don't
    lose rank either, since the field takes the same hit.
  - DIFFERENTIAL picks: low-ownership players. Risky - a blank costs you
    nothing extra, but a haul gains real rank because most rivals don't
    have them.

Which of these you should lean toward depends on your target: chasing rank
(behind where you want to be) rewards differentials; protecting rank (ahead
of target) rewards going template-heavy and cutting variance.

VALIDATION NOTE: the core "hidden gem" claim (low ownership + high predicted
points = worth it) is backtested below against real historical data - see
backtest_risk_analyzer.py. Everything else here (the tiering, the target-rank
framing) is a UI/decision-framing layer on top of data and predictions
that are already independently validated elsewhere in this repo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

OWNERSHIP_TIERS = [
    (0, 2, "Punt"),
    (2, 10, "Differential"),
    (10, 25, "Balanced"),
    (25, 200, "Template"),
]


def ownership_tier(pct: float) -> str:
    for low, high, label in OWNERSHIP_TIERS:
        if low <= pct < high:
            return label
    return "Template"


def analyze(players: pd.DataFrame, target_rank_direction: str = "chasing") -> pd.DataFrame:
    """players needs: id, web_name, position, now_cost, predicted_points,
    selected_by_percent (already present in gw1_seed_features.csv / live
    bootstrap-static data).

    target_rank_direction: 'chasing' (behind target, want variance/upside)
    or 'protecting' (ahead of target, want to minimize variance).

    IMPORTANT (see backtest_risk_analyzer.py): low ownership is NOT a free
    bonus. Backtested against real 2025-26 data, low-ownership players score
    ~1 point/gameweek WORSE than high-ownership players at the same price -
    ownership is a genuinely informative signal, not crowd noise. So this
    does NOT reward obscurity for its own sake. Instead it looks for players
    whose predicted_points rank is meaningfully HIGHER than their ownership
    rank within their price band - a real mismatch between (validated) model
    quality and market attention, not just "nobody's heard of them."
    """
    df = players.copy()
    ownership = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    df["ownership_pct"] = ownership
    df["tier"] = ownership.apply(ownership_tier)

    df["price_band"] = pd.cut(df["now_cost"], bins=[0, 50, 70, 90, 999], labels=["budget", "mid", "premium", "elite"])
    df["points_rank_in_band"] = df.groupby("price_band")["predicted_points"].rank(pct=True, ascending=True)
    df["ownership_rank_in_band"] = df.groupby("price_band")["ownership_pct"].rank(pct=True, ascending=True)

    # positive = model rates them higher than the market does = genuine
    # differential value; negative = market rates them higher than the model
    # does = likely NOT a real differential, just under-the-radar for a reason
    df["differential_score"] = df["points_rank_in_band"] - df["ownership_rank_in_band"]

    max_points = df["predicted_points"].max() if df["predicted_points"].max() > 0 else 1
    df["template_score"] = (df["predicted_points"] / max_points) * (ownership / 100).clip(upper=1)

    if target_rank_direction == "chasing":
        df["recommendation_score"] = df["differential_score"]
        strategy_note = ("Chasing rank: surfacing players the model rates higher than their ownership "
                          "suggests - real mismatches, not just low-ownership picks for their own sake.")
    else:
        df["recommendation_score"] = df["template_score"]
        strategy_note = "Protecting rank: leaning toward template picks - minimizing variance matters more than upside."

    df.attrs["strategy_note"] = strategy_note
    return df.sort_values("recommendation_score", ascending=False)


def top_picks_by_position(analyzed: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return analyzed.sort_values("recommendation_score", ascending=False).groupby("position").head(n)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from squad_optimizer import load_predictions

    preds = load_predictions()
    preds = preds[preds["available"]]

    for direction in ["chasing", "protecting"]:
        result = analyze(preds, target_rank_direction=direction)
        print(f"\n=== {result.attrs['strategy_note']} ===")
        top = top_picks_by_position(result, n=3)
        print(top[["web_name", "position", "now_cost", "predicted_points", "ownership_pct", "tier"]]
              .assign(now_cost=lambda d: (d["now_cost"] / 10).map("£{:.1f}m".format))
              .to_string(index=False))
