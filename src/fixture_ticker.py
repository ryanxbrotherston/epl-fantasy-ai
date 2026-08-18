"""
fixture_ticker.py — upcoming fixture difficulty, ranked per team.

Deliberately NOT a separate difficulty heuristic - reuses match_model.py's
fitted attack/defence lambdas directly, since those are already backtested
(45.5% match-winner accuracy, Brier 0.208 vs 0.218 naive - see README) rather
than an independent, unvalidated star-rating system.

Two difficulty angles, since they matter for different decisions:
  attacking_difficulty: how hard is it for THIS team to score against their
    upcoming opponents (relevant for captaincy / attacking picks)
  defensive_difficulty: how hard is it for THIS team to keep a clean sheet
    against their upcoming opponents (relevant for defenders/GK picks)
Both expressed as the opponent-adjusted expected goals for/against, scaled
against the league-average lambda so 1.0 = average difficulty, >1.0 = harder.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"


def build_ticker(fixtures: pd.DataFrame, model_bundle: dict, league_avg_lambda: float,
                  n_gameweeks: int = 5) -> pd.DataFrame:
    """fixtures needs columns: event (gameweek number), team_h_name, team_a_name.
    Call this with LIVE upcoming fixtures (from fpl_api.get_fixtures()) for
    real use, or historical fixtures for backtesting/demo.
    """
    from match_model import predict_lambda
    model, encoder = model_bundle["model"], model_bundle["encoder"]

    upcoming_gws = sorted(fixtures["event"].dropna().unique())[:n_gameweeks]
    fixtures = fixtures[fixtures["event"].isin(upcoming_gws)]

    rows = []
    for _, fx in fixtures.iterrows():
        home, away = fx["team_h_name"], fx["team_a_name"]

        home_attack_lambda = predict_lambda(model, encoder, home, away, 1)  # home's expected goals FOR
        away_attack_lambda = predict_lambda(model, encoder, away, home, 0)  # away's expected goals FOR

        if np.isnan(home_attack_lambda) or np.isnan(away_attack_lambda):
            continue  # promoted team with no fitted history yet

        rows.append({
            "event": fx["event"], "team": home, "opponent": away, "venue": "H",
            "attacking_difficulty": home_attack_lambda / league_avg_lambda,
            # this team's defensive difficulty = how much the OPPONENT is expected to score
            "defensive_difficulty": away_attack_lambda / league_avg_lambda,
        })
        rows.append({
            "event": fx["event"], "team": away, "opponent": home, "venue": "A",
            "attacking_difficulty": away_attack_lambda / league_avg_lambda,
            "defensive_difficulty": home_attack_lambda / league_avg_lambda,
        })

    return pd.DataFrame(rows)


def team_summary(ticker: pd.DataFrame) -> pd.DataFrame:
    """One row per team: average difficulty across their next N fixtures,
    plus the fixture list itself as a compact string for display."""
    def fixture_str(group):
        return ", ".join(f"{r.opponent}({r.venue})" for r in group.sort_values("event").itertuples())

    summary = ticker.groupby("team").agg(
        avg_attacking_difficulty=("attacking_difficulty", "mean"),
        avg_defensive_difficulty=("defensive_difficulty", "mean"),
        num_fixtures=("event", "count"),
    )
    summary["fixtures"] = ticker.groupby("team").apply(fixture_str, include_groups=False)
    return summary.sort_values("avg_defensive_difficulty")  # easiest defensive run first - good for defender/GK picks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from match_model import load_fixtures, TEST_SEASON
    from player_props import league_average_lambda

    # demo/self-test using historical fixtures (a live run would pass
    # fpl_api.get_fixtures() output instead)
    bundle = joblib.load(MODEL_DIR / "match_model.pkl")
    teams = pd.read_csv(BASE / "fpl-historical" / "data" / TEST_SEASON / "teams.csv")["name"].tolist()
    avg_lambda = league_average_lambda(bundle, teams)

    fixtures = load_fixtures(TEST_SEASON).rename(columns={"home_name": "team_h_name", "away_name": "team_a_name"})
    ticker = build_ticker(fixtures[fixtures["event"] <= 5], bundle, avg_lambda, n_gameweeks=5)
    summary = team_summary(ticker)

    print(f"Fixture difficulty ticker, first 5 gameweeks of {TEST_SEASON} (demo using historical data):\n")
    print("Easiest defensive runs (best for defenders/GK):")
    print(summary.head(5)[["avg_defensive_difficulty", "fixtures"]].to_string())
    print("\nHardest defensive runs:")
    print(summary.tail(5)[["avg_defensive_difficulty", "fixtures"]].to_string())
