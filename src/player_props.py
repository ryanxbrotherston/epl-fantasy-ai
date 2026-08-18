"""
player_props.py — anytime goalscorer / assist probabilities per fixture.

Combines two things we've already built rather than training a third model
from scratch:
  1. A player's own scoring rate (rolling expected-goals/assists per 90,
     same rolling features the points model uses - see feature_config.py)
  2. This specific fixture's difficulty, from match_model.py's lambda
     (expected team goals) relative to the league-average lambda

The player's own rate gets scaled up or down by how favorable this matchup
is, then treated as a Poisson rate for that single match - same simplifying
assumption match_model.py uses for team goals, same caveat applies (no
low-score correlation correction).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"

# Position-average goals/assists per 90, computed from TRAINING seasons only
# (2019-20 to 2024-25 - see src/backtest_player_props.py for how this was
# derived) - used as a shrinkage prior below. A player with a small, noisy
# rolling sample gets pulled toward this; a player with a large, established
# sample stays close to their own rate.
POSITION_PRIOR_PER90 = {
    "GK":  {"goals": 0.00007, "assists": 0.00076},
    "DEF": {"goals": 0.02288, "assists": 0.02493},
    "MID": {"goals": 0.07801, "assists": 0.05903},
    "FWD": {"goals": 0.17418, "assists": 0.03331},
}
# how many minutes of rolling history counts as "fully trust the player's own
# rate" - below this, blend toward the position prior. ~2 full matches.
SHRINKAGE_MINUTES_SCALE = 180


def player_rate_per90(roll_expected_stat: float, roll_minutes: float, fallback_roll_stat: float,
                        position: str, stat_type: str) -> float:
    """expected-goals/assists per 90 minutes, falling back to raw
    goals/assists rolling average when xG/xA isn't available (pre-2022-23
    seasons, or a player with too little data for xG to have stabilized),
    then SHRUNK toward the position-average rate in proportion to how little
    rolling minutes actually back the estimate - a player with one big game
    in a 5-game rolling window shouldn't have that game extrapolated to a
    full-strength 90-minute rate at face value."""
    if roll_minutes is None or roll_minutes <= 0:
        return 0.0
    rate_source = roll_expected_stat if (roll_expected_stat and roll_expected_stat > 0) else fallback_roll_stat
    own_rate = (rate_source or 0.0) / roll_minutes * 90

    prior = POSITION_PRIOR_PER90.get(position, POSITION_PRIOR_PER90["MID"])[stat_type]
    # shrinkage weight: 0 when roll_minutes is tiny, approaches 1 as it grows past SHRINKAGE_MINUTES_SCALE
    weight = roll_minutes / (roll_minutes + SHRINKAGE_MINUTES_SCALE)
    return weight * own_rate + (1 - weight) * prior


def fixture_adjusted_rate(player_rate_per90: float, expected_minutes: float,
                            team_lambda_this_fixture: float, league_avg_team_lambda: float) -> float:
    """Scale the player's own rate by how much stronger/weaker this specific
    matchup is than average - a player facing a leaky defence gets scaled up,
    a tough away fixture gets scaled down."""
    if league_avg_team_lambda <= 0:
        difficulty_multiplier = 1.0
    else:
        difficulty_multiplier = team_lambda_this_fixture / league_avg_team_lambda
    return player_rate_per90 * (expected_minutes / 90) * difficulty_multiplier


def anytime_probability(player_lambda_this_match: float) -> float:
    """P(at least one goal/assist) for a Poisson rate over one match."""
    return 1 - poisson.pmf(0, max(player_lambda_this_match, 0))


def player_fixture_props(player_row: pd.Series, team_lambda: float, league_avg_lambda: float,
                          expected_minutes: float | None = None) -> dict:
    """player_row needs: roll_expected_goals, roll_expected_assists, roll_minutes,
    roll_goals_scored, roll_assists, position (all already-computed rolling
    features, e.g. from build_gw1_features.py's output or a live in-season
    equivalent). expected_minutes defaults to the player's own rolling minutes
    average if not given explicitly - a flat assumption (e.g. always 75) was
    tested and found to badly overstate props for fringe/rotation players;
    see backtest_player_props.py."""
    roll_minutes = player_row.get("roll_minutes") or 0
    if expected_minutes is None:
        expected_minutes = min(roll_minutes, 90)  # can't play more than a full match regardless of rolling average

    position = player_row.get("position", "MID")

    goal_rate90 = player_rate_per90(
        player_row.get("roll_expected_goals"), roll_minutes, player_row.get("roll_goals_scored"),
        position, "goals",
    )
    assist_rate90 = player_rate_per90(
        player_row.get("roll_expected_assists"), roll_minutes, player_row.get("roll_assists"),
        position, "assists",
    )

    goal_lambda = fixture_adjusted_rate(goal_rate90, expected_minutes, team_lambda, league_avg_lambda)
    assist_lambda = fixture_adjusted_rate(assist_rate90, expected_minutes, team_lambda, league_avg_lambda)

    return {
        "p_scores": anytime_probability(goal_lambda),
        "p_assists": anytime_probability(assist_lambda),
        "expected_goals_this_match": goal_lambda,
        "expected_assists_this_match": assist_lambda,
    }


def league_average_lambda(match_model_bundle: dict, all_teams: list[str]) -> float:
    """Baseline used to judge whether a specific fixture is 'easy' or 'hard' -
    the average expected-goals rate across all team/opponent combinations in
    training, home and away combined."""
    from match_model import predict_lambda
    model, encoder = match_model_bundle["model"], match_model_bundle["encoder"]
    lambdas = []
    for team in all_teams:
        for opp in all_teams:
            if team == opp:
                continue
            for is_home in (0, 1):
                lam = predict_lambda(model, encoder, team, opp, is_home)
                if not np.isnan(lam):
                    lambdas.append(lam)
    return float(np.mean(lambdas)) if lambdas else 1.4  # ~1.4 goals/team/match is a typical PL baseline
