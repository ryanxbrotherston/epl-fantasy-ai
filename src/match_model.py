"""
match_model.py — Poisson goal model for match outcome predictions.

Standard approach in football analytics (the "Maher model", refined by
Dixon-Coles 1997): each team has an attack strength and a defence strength.
Expected goals for a fixture = league baseline × attacking team's attack
strength × defending team's (in)ability to keep them out. Fit as a Poisson
regression on a "long format" of every historical match (two rows per match:
one for each team's scoring output), which lets attack and defence strength
fall out as regression coefficients rather than needing separate models.

Everything else — match winner, exact scorelines, clean sheets — is derived
mathematically from the two fitted expected-goals numbers (lambda_home,
lambda_away) for a fixture, assuming goals are Poisson-distributed and
(as a known simplification - see backtest notes) independent between teams.
The real Dixon-Coles model adds a small correlation correction for low
scores (0-0, 1-0, 0-1, 1-1) that this doesn't implement yet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"
MODEL_DIR = BASE / "models"

TEST_SEASON = "2025-26"
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
ALL_SEASONS_FOR_DECAY = TRAIN_SEASONS + [TEST_SEASON]
MAX_GOALS_GRID = 10  # truncation point for summing the Poisson grid - probabilities beyond this are negligible


def load_fixtures(season: str) -> pd.DataFrame:
    fixtures = pd.read_csv(DATA_DIR / season / "fixtures.csv", encoding="utf-8-sig")
    teams = pd.read_csv(DATA_DIR / season / "teams.csv", encoding="utf-8-sig")[["id", "name"]]

    fixtures = fixtures[fixtures["finished"] == True].copy()
    fixtures = fixtures.merge(teams.rename(columns={"id": "team_h", "name": "home_name"}), on="team_h")
    fixtures = fixtures.merge(teams.rename(columns={"id": "team_a", "name": "away_name"}), on="team_a")
    fixtures["season"] = season
    return fixtures[["season", "event", "home_name", "away_name", "team_h_score", "team_a_score"]]


def to_long_format(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one for each team's goal-scoring output, with
    who they played and whether they were at home. This is what lets a
    single Poisson regression estimate attack AND defence strength at once -
    the opponent's ability to prevent goals shows up as an opponent effect."""
    home_rows = fixtures.rename(columns={
        "home_name": "team", "away_name": "opponent", "team_h_score": "goals",
    })[["season", "team", "opponent", "goals"]]
    home_rows["is_home"] = 1

    away_rows = fixtures.rename(columns={
        "away_name": "team", "home_name": "opponent", "team_a_score": "goals",
    })[["season", "team", "opponent", "goals"]]
    away_rows["is_home"] = 0

    return pd.concat([home_rows, away_rows], ignore_index=True)


def fit_model(long_df: pd.DataFrame, decay: float = 0.7):
    season_order = {s: i for i, s in enumerate(ALL_SEASONS_FOR_DECAY)}
    test_idx = season_order[TEST_SEASON]
    sample_weight = long_df["season"].map(lambda s: decay ** (test_idx - season_order[s])).values

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    team_opp = encoder.fit_transform(long_df[["team", "opponent"]])
    X = np.hstack([team_opp, long_df[["is_home"]].values])

    model = PoissonRegressor(alpha=1e-3, max_iter=500)
    model.fit(X, long_df["goals"], sample_weight=sample_weight)
    return model, encoder


def predict_lambda(model, encoder, team: str, opponent: str, is_home: int) -> float:
    row = pd.DataFrame([{"team": team, "opponent": opponent}])
    try:
        team_opp = encoder.transform(row[["team", "opponent"]])
    except ValueError:
        return np.nan  # unseen team (e.g. newly promoted with no history yet)
    X = np.hstack([team_opp, [[is_home]]])
    return float(model.predict(X)[0])


def fixture_probabilities(lambda_home: float, lambda_away: float) -> dict:
    """Everything derived from the two Poisson rates for one fixture."""
    goals = np.arange(0, MAX_GOALS_GRID + 1)
    p_home_goals = poisson.pmf(goals, lambda_home)
    p_away_goals = poisson.pmf(goals, lambda_away)
    score_grid = np.outer(p_home_goals, p_away_goals)  # [i, j] = P(home scores i, away scores j)

    p_home_win = np.tril(score_grid, k=-1).sum()
    p_draw = np.trace(score_grid)
    p_away_win = np.triu(score_grid, k=1).sum()

    top_scores_idx = np.dstack(np.unravel_index(np.argsort(-score_grid.ravel()), score_grid.shape))[0][:5]
    top_scores = [{"score": f"{i}-{j}", "prob": float(score_grid[i, j])} for i, j in top_scores_idx]

    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away,
        "p_home_win": float(p_home_win), "p_draw": float(p_draw), "p_away_win": float(p_away_win),
        "p_home_clean_sheet": float(poisson.pmf(0, lambda_away)),
        "p_away_clean_sheet": float(poisson.pmf(0, lambda_home)),
        "top_scores": top_scores,
    }


def train_and_save():
    train_fixtures = pd.concat([load_fixtures(s) for s in TRAIN_SEASONS], ignore_index=True)
    long_df = to_long_format(train_fixtures)
    model, encoder = fit_model(long_df)

    import joblib
    joblib.dump({"model": model, "encoder": encoder}, MODEL_DIR / "match_model.pkl")
    print(f"Trained on {len(train_fixtures)} matches across {TRAIN_SEASONS}. Saved to models/match_model.pkl")
    return model, encoder


if __name__ == "__main__":
    train_and_save()
