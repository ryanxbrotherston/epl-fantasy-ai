"""
live_rolling_features.py — this season's actual rolling player form, pulled
live, replacing the frozen last-season snapshot from build_gw1_features.py.

Uses element-summary/{player_id}/'s 'history' field (confirmed via multiple
independent doc sources - see conversation history - this endpoint returns
fixtures/history/history_past; 'history' is this season's gameweek-by-
gameweek data for that player). Computes the SAME rolling features
(feature_config.CORE_STATS/OPTIONAL_STATS, same window) as training, so the
trained model can be applied to genuinely current in-season form rather than
a pre-season proxy.

Untested against the live API for the same reason as fpl_api.py - this
sandbox can't reach fantasy.premierleague.com. Logic is validated below
against schema-accurate mock data. Confirm against a real gameweek once
GW1+ has actually been played.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fpl_api
from feature_config import ROLLING_WINDOW, CORE_STATS, OPTIONAL_STATS


def player_rolling_form(player_id: int, as_of_event: int | None = None) -> dict | None:
    """This player's rolling-mean stats over their last ROLLING_WINDOW played
    gameweeks THIS season, as of a given event (defaults to their most recent
    played gameweek). Returns None if they have no gameweek history yet
    (hasn't played this season, e.g. very start of GW1)."""
    summary = fpl_api.get_element_summary(player_id)
    if summary is None or not summary.get("history"):
        return None

    history = pd.DataFrame(summary["history"])
    if as_of_event is not None:
        history = history[history["round"] <= as_of_event]
    if history.empty:
        return None

    history = history.sort_values("round")
    stat_cols = [c for c in CORE_STATS + OPTIONAL_STATS if c in history.columns]
    # FPL's live API returns some stats (ict_index, threat, creativity, influence)
    # as quoted strings, not numbers - coerce before any arithmetic
    for c in stat_cols:
        history[c] = pd.to_numeric(history[c], errors="coerce")
    recent = history.tail(ROLLING_WINDOW)

    result = {f"roll_{c}": recent[c].mean() for c in stat_cols}
    # zero-fill any stat this endpoint doesn't carry (schema can vary slightly
    # from the historical CSVs) so downstream code always sees every expected column
    for c in CORE_STATS + OPTIONAL_STATS:
        result.setdefault(f"roll_{c}", 0.0)

    result["games_played_so_far"] = len(history)
    result["player_id"] = player_id
    return result


def build_live_squad_features(player_ids: list[int], bootstrap: dict) -> pd.DataFrame:
    """This season's live rolling features for a list of players (e.g. an
    entire squad, or all 500+ players for a full-league recompute), falling
    back to build_gw1_features.py's pre-season proxy for anyone with no
    in-season history yet (new signings mid-season, or very early GW1)."""
    from build_gw1_features import build as build_preseason_proxy

    live_rows = []
    for pid in player_ids:
        form = player_rolling_form(pid)
        if form is not None:
            live_rows.append(form)

    live_df = pd.DataFrame(live_rows) if live_rows else pd.DataFrame(columns=["player_id"])

    proxy_df = build_preseason_proxy()
    proxy_df = proxy_df.rename(columns={"id": "player_id"})

    # live data wins where we have it; pre-season proxy fills the rest
    combined = proxy_df.set_index("player_id")
    if not live_df.empty:
        combined.update(live_df.set_index("player_id"))
    return combined.reset_index()


if __name__ == "__main__":
    # self-test against schema-accurate mock data - mirrors the pattern used
    # to validate fpl_api.py, since live calls aren't reachable from here
    import types

    def mock_get_element_summary(player_id):
        return {
            "history": [
                {"round": 1, "minutes": 90, "total_points": 6, "goals_scored": 1, "assists": 0,
                 "clean_sheets": 0, "goals_conceded": 1, "bonus": 2, "bps": 30, "ict_index": "8.5",
                 "threat": "40", "creativity": "20", "influence": "35", "yellow_cards": 0, "red_cards": 0,
                 "saves": 0, "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0},
                {"round": 2, "minutes": 90, "total_points": 2, "goals_scored": 0, "assists": 0,
                 "clean_sheets": 0, "goals_conceded": 2, "bonus": 0, "bps": 15, "ict_index": "4.0",
                 "threat": "15", "creativity": "10", "influence": "20", "yellow_cards": 1, "red_cards": 0,
                 "saves": 0, "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0},
            ],
            "fixtures": [], "history_past": [],
        }

    fpl_api.get_element_summary = mock_get_element_summary
    form = player_rolling_form(player_id=1)
    assert form is not None, "FAIL: expected rolling form, got None"
    assert abs(form["roll_total_points"] - 4.0) < 1e-9, f"FAIL: expected mean(6,2)=4.0, got {form['roll_total_points']}"
    assert form["games_played_so_far"] == 2
    print("Mock schema test passed:", {k: v for k, v in form.items() if k in ("roll_total_points", "roll_minutes", "games_played_so_far")})

    # empty-history case
    fpl_api.get_element_summary = lambda pid: {"history": [], "fixtures": [], "history_past": []}
    assert player_rolling_form(player_id=2) is None, "FAIL: expected None for a player with no history yet"
    print("Empty-history edge case handled correctly")
    print("\nALL SELF-TESTS PASSED (mock data only - confirm against a real live gameweek before trusting this)")
