"""
build_gw1_features.py — Seed GW1 predictions with last season's closing form.

Problem: our rolling-window model needs trailing gameweek history, but GW1
of a new season has none *this* season. Fix: pull each returning player's
final-N-gameweek rolling stats from the END of last season (2025-26) as the
seed, matched onto this season's (2026-27) current squad list, prices and
teams. Players with no match (new signings, promoted-team players, kids
coming through) get a position + price-tier fallback instead of a hard fail.
"""

import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_config import ROLLING_WINDOW, CORE_STATS, OPTIONAL_STATS, POSITION_ID_MAP

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "fpl-historical" / "data"
MODEL_DIR = BASE / "models"


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z ]", "", s).lower().strip()
    return s


def last_season_closing_form(season="2025-26") -> pd.DataFrame:
    """Each player's rolling-mean stats as of the LAST gameweek they played
    last season - i.e. their closing form, which becomes the seed for GW1."""
    df = pd.read_csv(DATA_DIR / season / "gws" / "merged_gw.csv", encoding="utf-8-sig")
    for col in OPTIONAL_STATS:
        if col not in df.columns:
            df[col] = 0
    df = df.sort_values(["element", "GW"])

    grouped = df.groupby("element", group_keys=False)
    rolling = {}
    for col in CORE_STATS + OPTIONAL_STATS:
        rolling[f"roll_{col}"] = grouped[col].apply(
            lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
    rolled = pd.concat(rolling, axis=1)
    df = pd.concat([df[["element", "name", "GW"]], rolled], axis=1)

    # take each player's LAST row (their season-closing rolling form)
    closing = df.sort_values("GW").groupby("element").tail(1).reset_index(drop=True)
    closing["norm_name"] = closing["name"].apply(normalize_name)
    closing["games_played_so_far"] = 38  # full season completed - max recency signal
    return closing


CURRENT_SEASON_SIGNAL_COLS = [
    "ep_next", "ep_this", "total_points", "points_per_game", "form",
    "status", "chance_of_playing_next_round", "chance_of_playing_this_round",
    "selected_by_percent", "news",
]


def current_squad(season="2026-27") -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / season / "players_raw.csv", encoding="utf-8-sig")
    df["full_name"] = df["first_name"].fillna("") + " " + df["second_name"].fillna("")
    df["norm_name"] = df["full_name"].apply(normalize_name)
    df["norm_web_name"] = df["web_name"].apply(normalize_name)
    df["position"] = df["element_type"].map(POSITION_ID_MAP)
    for col in CURRENT_SEASON_SIGNAL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build():
    closing = last_season_closing_form()
    squad = current_squad()

    feature_cols_all = [f"roll_{c}" for c in CORE_STATS + OPTIONAL_STATS] + ["games_played_so_far"]

    # lookup dict keyed by normalized full name, with normalized last-name-only
    # fallback (web_name usually IS the surname), tried in that order
    by_full = closing.set_index("norm_name")[feature_cols_all].to_dict("index")

    def lookup(row):
        for key in (row["norm_name"], row["norm_web_name"]):
            if key in by_full:
                return by_full[key]
        return None

    matches = squad.apply(lookup, axis=1)
    matched_mask = matches.notna()
    for col in feature_cols_all:
        squad[col] = matches.apply(lambda m: m[col] if m is not None else np.nan)
    merged = squad
    print(f"Matched {matched_mask.sum()} / {len(merged)} current players to last season's closing form")

    # fallback for unmatched: position + price-tier average of MATCHED players' rolling stats
    feature_cols = [f"roll_{c}" for c in CORE_STATS + OPTIONAL_STATS]
    merged["price_tier"] = pd.cut(merged["now_cost"], bins=[0, 45, 55, 70, 90, 999], labels=["budget", "low", "mid", "premium", "elite"])

    tier_means = merged[matched_mask].groupby(["position", "price_tier"], observed=True)[feature_cols + ["games_played_so_far"]].mean()
    overall_pos_means = merged[matched_mask].groupby("position")[feature_cols + ["games_played_so_far"]].mean()

    for idx in merged[~matched_mask].index:
        pos, tier = merged.loc[idx, "position"], merged.loc[idx, "price_tier"]
        if (pos, tier) in tier_means.index:
            merged.loc[idx, feature_cols + ["games_played_so_far"]] = tier_means.loc[(pos, tier)].values
        elif pos in overall_pos_means.index:
            merged.loc[idx, feature_cols + ["games_played_so_far"]] = overall_pos_means.loc[pos].values
        else:
            merged.loc[idx, feature_cols + ["games_played_so_far"]] = 0
        merged.loc[idx, "games_played_so_far"] = 0  # they have zero games THIS season/club regardless

    out_cols = (["id", "web_name", "full_name", "position", "team", "now_cost"]
                + feature_cols + ["games_played_so_far"] + CURRENT_SEASON_SIGNAL_COLS)
    result = merged[out_cols].copy()
    result["matched_last_season"] = matched_mask.values
    return result


if __name__ == "__main__":
    result = build()
    out_path = BASE / "data" / "gw1_seed_features.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} players to {out_path}")
    print(result[["web_name", "position", "now_cost", "roll_total_points", "matched_last_season"]].sort_values("roll_total_points", ascending=False).head(10))
