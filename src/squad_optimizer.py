"""
squad_optimizer.py — Initial squad selection via integer linear programming.

Applies the trained points model to the GW1 seed features, then solves:
maximize total predicted points, subject to:
  - exactly 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
  - total cost <= budget (1000 = £100.0m, FPL's tenths-of-a-million units)
  - at most 3 players from any one real-world club
Also proposes a starting XI (valid formation, best 11 of the 15) and captain.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pulp

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"

BUDGET = 1000       # £100.0m
SQUAD_SIZE = 15
POSITION_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
TEAM_LIMIT = 3

# valid starting XI formations (GK is always 1): (DEF, MID, FWD)
VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1), (5, 2, 3),
]


# Blend weights for the final predicted-points signal used by the optimizer.
# roll_model: our rolling-form RandomForest (data-driven, backtested MAE 0.99)
# ep_next:    FPL's own next-gameweek estimate (accounts for known fixture,
#             set-piece duty, current price/form - things our model can't see)
# ppg:        last season's points-per-game (proven quality anchor, same
#             per-gameweek scale as the other two signals)
BLEND_WEIGHTS = {"model": 0.40, "ep_next": 0.40, "ppg": 0.20}

# statuses FPL uses: a=available, d=doubtful, i=injured, s=suspended, u=unavailable/left club
UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}


def load_predictions() -> pd.DataFrame:
    model = joblib.load(MODEL_DIR / "points_model.pkl")
    with open(MODEL_DIR / "feature_columns.json") as f:
        meta = json.load(f)

    df = pd.read_csv(DATA_DIR / "gw1_seed_features.csv")
    df_feat = pd.get_dummies(df, columns=["position"], prefix="pos")
    for p in ["GK", "DEF", "MID", "FWD"]:
        col = f"pos_{p}"
        if col not in df_feat.columns:
            df_feat[col] = 0

    X = df_feat[meta["features"]]
    df["model_points"] = model.predict(X)

    ep_next = pd.to_numeric(df["ep_next"], errors="coerce").fillna(df["model_points"])
    ppg = pd.to_numeric(df["points_per_game"], errors="coerce").fillna(df["model_points"])

    df["predicted_points"] = (
        BLEND_WEIGHTS["model"] * df["model_points"]
        + BLEND_WEIGHTS["ep_next"] * ep_next
        + BLEND_WEIGHTS["ppg"] * ppg
    )

    # doubtful players: discount by their listed chance of playing
    chance = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce")
    doubtful = df["status"].eq("d") & chance.notna()
    df.loc[doubtful, "predicted_points"] *= (chance[doubtful] / 100)

    df["available"] = ~df["status"].isin(UNAVAILABLE_STATUSES)
    return df


def pick_squad(df: pd.DataFrame, budget: int = BUDGET, excluded_ids=None, locked_ids=None) -> pd.DataFrame:
    """Solve the 15-man squad ILP. excluded_ids: players to never pick
    (e.g. injured/unavailable). locked_ids: players forced into the squad."""
    excluded_ids = excluded_ids or set()
    locked_ids = locked_ids or set()
    pool = df[df["available"] & ~df["id"].isin(excluded_ids)].reset_index(drop=True)

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}

    prob += pulp.lpSum(x[i] * pool.loc[i, "predicted_points"] for i in pool.index)

    prob += pulp.lpSum(x[i] for i in pool.index) == SQUAD_SIZE
    prob += pulp.lpSum(x[i] * pool.loc[i, "now_cost"] for i in pool.index) <= budget

    for pos, quota in POSITION_QUOTAS.items():
        prob += pulp.lpSum(x[i] for i in pool.index if pool.loc[i, "position"] == pos) == quota

    for team_id in pool["team"].unique():
        prob += pulp.lpSum(x[i] for i in pool.index if pool.loc[i, "team"] == team_id) <= TEAM_LIMIT

    for i in pool.index:
        if pool.loc[i, "id"] in locked_ids:
            prob += x[i] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")

    chosen = [i for i in pool.index if x[i].value() == 1]
    return pool.loc[chosen].sort_values(["position", "predicted_points"], ascending=[True, False])


def pick_starting_xi(squad: pd.DataFrame):
    """Given the 15-man squad, pick the best-scoring valid starting XI + captain."""
    best_xi, best_score, best_formation = None, -1, None
    for def_n, mid_n, fwd_n in VALID_FORMATIONS:
        gk = squad[squad.position == "GK"].nlargest(1, "predicted_points")
        de = squad[squad.position == "DEF"].nlargest(def_n, "predicted_points")
        mi = squad[squad.position == "MID"].nlargest(mid_n, "predicted_points")
        fw = squad[squad.position == "FWD"].nlargest(fwd_n, "predicted_points")
        if len(de) < def_n or len(mi) < mid_n or len(fw) < fwd_n:
            continue
        xi = pd.concat([gk, de, mi, fw])
        score = xi["predicted_points"].sum()
        if score > best_score:
            best_xi, best_score, best_formation = xi, score, f"{def_n}-{mid_n}-{fwd_n}"

    bench = squad[~squad["id"].isin(best_xi["id"])].sort_values("predicted_points", ascending=False)
    captain = best_xi.nlargest(1, "predicted_points").iloc[0]
    vice = best_xi.drop(captain.name).nlargest(1, "predicted_points").iloc[0]
    return best_xi, bench, best_formation, captain, vice


if __name__ == "__main__":
    preds = load_predictions()
    squad = pick_squad(preds)
    xi, bench, formation, captain, vice = pick_starting_xi(squad)

    spend = squad["now_cost"].sum() / 10
    print(f"=== 15-man squad (£{spend:.1f}m / £100.0m) ===")
    print(squad[["web_name", "position", "team", "now_cost", "predicted_points", "matched_last_season"]]
          .assign(now_cost=lambda d: d["now_cost"] / 10)
          .to_string(index=False))

    print(f"\n=== Starting XI ({formation}) — predicted {xi['predicted_points'].sum():.1f} pts ===")
    print(xi[["web_name", "position", "predicted_points"]].to_string(index=False))
    print(f"\nCaptain: {captain['web_name']} ({captain['predicted_points']:.1f} pred pts)")
    print(f"Vice-captain: {vice['web_name']} ({vice['predicted_points']:.1f} pred pts)")
    print(f"\nBench: {', '.join(bench['web_name'].tolist())}")
