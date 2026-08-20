"""
multi_week_projections.py — predicted points per player, per upcoming
gameweek, fixture-adjusted (and correctly handling double/blank gameweeks -
sum multiple fixtures, zero for none).

This is what makes the multi-week planner's chip timing genuinely dynamic:
a double gameweek shows up as elevated projected points for players on the
teams involved, a blank shows up as zero - the planner responds to whatever
this table says, nothing about chip timing is hardcoded.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"


def build_gw_fixture_map(fixtures: pd.DataFrame, gameweeks: list[int]) -> dict:
    """{team_name: {gw: [(opponent_name, is_home), ...]}} - a list because a
    team can have 0, 1, or 2+ entries in a given gameweek."""
    fixture_map = {}
    for _, fx in fixtures[fixtures["event"].isin(gameweeks)].iterrows():
        gw = int(fx["event"])
        home, away = fx["team_h_name"], fx["team_a_name"]
        fixture_map.setdefault(home, {}).setdefault(gw, []).append((away, True))
        fixture_map.setdefault(away, {}).setdefault(gw, []).append((home, False))
    return fixture_map


def build_projection_table(squad_predictions: pd.DataFrame, fixtures: pd.DataFrame,
                             gameweeks: list[int], model_bundle: dict, league_avg_lambda: float,
                             team_name_col: str = "team_name") -> pd.DataFrame:
    """squad_predictions needs: id, web_name, position, team_name, now_cost,
    predicted_points (the season-baseline blended prediction, e.g. from
    squad_optimizer.load_predictions()). now_cost is carried through
    unmodified (not fixture-adjusted, unlike predicted_points) - the
    planner's budget constraint needs it on this table since it's the one
    passed into multi_week_planner.plan().
    Returns a wide table: one row per player, one column per gameweek.
    """
    from match_model import predict_lambda
    model, encoder = model_bundle["model"], model_bundle["encoder"]

    fixture_map = build_gw_fixture_map(fixtures, gameweeks)

    result = squad_predictions[["id", "web_name", "position", team_name_col, "now_cost", "predicted_points"]].copy()
    for gw in gameweeks:
        col = []
        for _, player in result.iterrows():
            team = player[team_name_col]
            team_fixtures = fixture_map.get(team, {}).get(gw, [])
            if not team_fixtures:
                col.append(0.0)
                continue

            gw_total = 0.0
            for opponent, is_home in team_fixtures:
                lam = predict_lambda(model, encoder, team, opponent, int(is_home))
                if np.isnan(lam):
                    gw_total += player["predicted_points"]  # unseen opponent - no adjustment, use baseline
                else:
                    difficulty_multiplier = lam / league_avg_lambda if league_avg_lambda > 0 else 1.0
                    gw_total += player["predicted_points"] * difficulty_multiplier
            col.append(gw_total)
        result[f"gw{gw}"] = col

    return result
