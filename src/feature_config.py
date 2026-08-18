"""
feature_config.py — single source of truth for feature lists, shared by
train_points_model.py and build_gw1_features.py so they can never silently
drift out of sync with each other.

CORE_STATS: verified present in every season 2016-17 through 2025-26.
OPTIONAL_STATS: only present in some seasons (availability varies - see
comments). Zero-filled where missing. RandomForest handles this reasonably
(it can learn "this feature is uninformative pre-2022" from the pattern),
but it's a real compromise, not free lunch - a season where a stat is
zero-filled isn't the same as a season where a player genuinely had zero.
"""

ROLLING_WINDOW = 5

CORE_STATS = [
    "minutes", "total_points", "ict_index", "threat", "creativity", "influence",
    "bps", "goals_scored", "assists", "clean_sheets", "goals_conceded", "bonus",
    "yellow_cards", "red_cards", "saves", "own_goals", "penalties_missed", "penalties_saved",
]

OPTIONAL_STATS = [
    "starts",                                                        # 2022-23 onward
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",          # 2022-23 onward
    "defensive_contribution",                                        # 2025-26 onward (the new scoring rule itself)
    "tackles", "clearances_blocks_interceptions", "recoveries",       # 2016-17 to 2018-19, THEN a gap, THEN back 2025-26 onward
]

POSITION_ID_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITIONS = ["GK", "DEF", "MID", "FWD"]

# seasons where merged_gw.csv itself doesn't carry position/team - must be
# joined in from that season's own players_raw.csv (element_type/team cols)
SEASONS_NEEDING_POSITION_JOIN = {"2016-17", "2017-18", "2018-19", "2019-20"}

ALL_HISTORICAL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
