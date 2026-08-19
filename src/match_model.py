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
lambda_away) for a fixture, assuming goals are Poisson-distributed. The full
Dixon-Coles correction adds a small correlation term (rho) that nudges the
four low-score cells (0-0, 1-0, 0-1, 1-1) away from what independent Poisson
alone predicts - real football has slightly more 0-0/1-1 draws and slightly
fewer 1-0/0-1 results than pure independence implies, because a team already
chasing/protecting a scrappy low score changes its own attacking intent in a
way a fixed lambda can't capture. rho is fit by MLE on the training seasons
(fit_dixon_coles_rho, holding lambda_home/lambda_away fixed from the already-
fitted Poisson regression) and applied in fixture_probabilities. See
backtest_match_model.py for the with/without comparison on held-out data.
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


def dixon_coles_tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """The Dixon-Coles (1997) low-score correction factor - only the four
    cells where either team scored 0 or 1 get adjusted; every other
    scoreline keeps its plain independent-Poisson probability (tau=1)."""
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles_rho(fixtures: pd.DataFrame, model, encoder, bounds=(-0.3, 0.3)) -> float:
    """MLE fit of rho, holding lambda_home/lambda_away fixed from the
    already-fitted Poisson regression - rho only touches 4 score cells, so
    it's a well-behaved 1-D search rather than something that needs
    refitting jointly with the attack/defence strengths."""
    from scipy.optimize import minimize_scalar

    observed = []
    for _, fx in fixtures.iterrows():
        lh = predict_lambda(model, encoder, fx["home_name"], fx["away_name"], 1)
        la = predict_lambda(model, encoder, fx["away_name"], fx["home_name"], 0)
        if np.isnan(lh) or np.isnan(la):
            continue
        observed.append((lh, la, int(fx["team_h_score"]), int(fx["team_a_score"])))

    def neg_log_lik(rho):
        total = 0.0
        for lh, la, hs, as_ in observed:
            tau = dixon_coles_tau(hs, as_, lh, la, rho)
            p = tau * poisson.pmf(hs, lh) * poisson.pmf(as_, la)
            if tau <= 0 or p <= 0:
                return 1e10  # outside the region where tau keeps probabilities valid
            total += np.log(p)
        return -total

    result = minimize_scalar(neg_log_lik, bounds=bounds, method="bounded")
    return float(result.x)


def fixture_probabilities(lambda_home: float, lambda_away: float, rho: float = 0.0) -> dict:
    """Everything derived from the two Poisson rates for one fixture, with
    an optional Dixon-Coles low-score correction (rho=0.0 reproduces plain
    independent Poisson)."""
    goals = np.arange(0, MAX_GOALS_GRID + 1)
    p_home_goals = poisson.pmf(goals, lambda_home)
    p_away_goals = poisson.pmf(goals, lambda_away)
    score_grid = np.outer(p_home_goals, p_away_goals)  # [i, j] = P(home scores i, away scores j)

    if rho != 0.0:
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            score_grid[x, y] *= dixon_coles_tau(x, y, lambda_home, lambda_away, rho)
        score_grid /= score_grid.sum()  # renormalize - tau perturbs 4 cells, grid should still sum to ~1

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
    rho = fit_dixon_coles_rho(train_fixtures, model, encoder)

    import joblib
    joblib.dump({"model": model, "encoder": encoder, "rho": rho}, MODEL_DIR / "match_model.pkl")
    print(f"Trained on {len(train_fixtures)} matches across {TRAIN_SEASONS}. "
          f"Dixon-Coles rho={rho:.4f}. Saved to models/match_model.pkl")
    return model, encoder, rho


if __name__ == "__main__":
    train_and_save()
