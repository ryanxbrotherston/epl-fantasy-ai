"""
fpl_api.py — Live FPL API connector.

The FPL API (fantasy.premierleague.com/api) is free, read-only, and needs no
auth for anything here — this is the same public data mini-leagues use to
show everyone else's teams, so pulling a friend's Team ID is exactly what
it's designed for.

NOTE: this sandbox's network allowlist doesn't include fantasy.premierleague.com,
so these calls are untested from within the build environment itself. The
schema is corroborated across multiple independent sources (FPL Squid, Oliver
Looney's FPL API guide, the fpl-mcp server, the `fpl` PyPI package) but do a
sanity check against your own real Team ID the first time you run this live.
"""

import requests

BASE = "https://fantasy.premierleague.com/api"
TIMEOUT = 10


def _get(path: str) -> dict | None:
    """GET a path, returning None on 404 (e.g. picks for a gameweek that
    hasn't happened yet) rather than raising, since that's an expected state
    pre-season / pre-deadline, not a real error."""
    try:
        resp = requests.get(f"{BASE}/{path}", timeout=TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"FPL API request failed for {path}: {e}") from e


def get_bootstrap_static() -> dict:
    """Everything: all players, all teams, all gameweeks, scoring rules,
    chip windows. Cache this for ~1 hour - it's the workhorse call."""
    return _get("bootstrap-static/")


def get_fixtures(event: int | None = None) -> list[dict]:
    path = "fixtures/" if event is None else f"fixtures/?event={event}"
    return _get(path) or []


def get_entry(team_id: int) -> dict | None:
    """Basic manager info: team name, manager name, overall points/rank,
    current gameweek. Returns None if the Team ID doesn't exist."""
    return _get(f"entry/{team_id}/")


def get_entry_picks(team_id: int, event: int) -> dict | None:
    """The 15 players picked for a specific gameweek, captain/vice, chip
    used, and that gameweek's points/rank/bank/value. Returns None before
    that gameweek's results are in (e.g. asking for GW1 before deadline)."""
    return _get(f"entry/{team_id}/event/{event}/picks/")


def get_entry_history(team_id: int) -> dict | None:
    """Full season history (every gameweek so far) plus past-season summaries
    and chip usage log for this manager."""
    return _get(f"entry/{team_id}/history/")


def get_entry_transfers(team_id: int) -> list[dict]:
    return _get(f"entry/{team_id}/transfers/") or []


def get_event_live(event: int) -> dict | None:
    """Every player's stats for ONE specific gameweek (minutes, points,
    bps, etc.) in a single request - much cheaper than calling
    get_element_summary() per player when you need a whole gameweek's
    worth of actuals at once. Returns None before that gameweek exists."""
    return _get(f"event/{event}/live/")


def get_element_summary(player_id: int) -> dict | None:
    """A single player's full fixture-by-fixture history this season plus
    upcoming fixtures - useful for the transfer-advice engine later."""
    return _get(f"element-summary/{player_id}/")


def current_gameweek(bootstrap: dict) -> dict:
    """Whichever event has is_current=True, or the next one if the season
    hasn't started (is_next=True) - GW1 pre-deadline falls in this second case."""
    events = bootstrap["events"]
    current = next((e for e in events if e["is_current"]), None)
    if current:
        return current
    return next(e for e in events if e["is_next"])


def players_dataframe(bootstrap: dict):
    """bootstrap['elements'] as a DataFrame with human-readable position and
    team name columns joined in, ready for display/merging."""
    import pandas as pd

    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]].rename(
        columns={"id": "team", "name": "team_name", "short_name": "team_short"}
    )
    # normalized to match squad_optimizer's labels (GK, not FPL's own "GKP")
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    elements["position"] = elements["element_type"].map(pos_map)
    elements = elements.merge(teams, on="team", how="left")
    return elements


def team_picks_dataframe(team_id: int, event: int, bootstrap: dict):
    """A friend's/your squad for a gameweek, joined with player names/prices/
    predicted-relevant stats. Returns None if that gameweek's picks aren't
    available yet (not played, or invalid Team ID)."""
    import pandas as pd

    picks_data = get_entry_picks(team_id, event)
    if picks_data is None:
        return None

    players = players_dataframe(bootstrap)
    picks = pd.DataFrame(picks_data["picks"])
    # picks_data["picks"] carries its own "position" field (squad slot
    # number 1-15, an int - not used anywhere in this codebase) which
    # collides with players_dataframe()'s "position" (GK/DEF/MID/FWD,
    # what every caller here actually means by that name). Left unrenamed,
    # pandas silently resolves the collision into position_x/position_y
    # instead of raising - the merged frame then has NO "position" column
    # at all, and every caller reading merged["position"] KeyErrors.
    # Confirmed live against a real GW1 picks response (2026-08-22, the
    # first time this endpoint returned real picks - this bug predates
    # this change and could never surface until a live picks response
    # actually existed to merge against).
    picks = picks.rename(columns={"position": "squad_slot"})
    merged = picks.merge(
        players[["id", "web_name", "position", "team", "team_short", "now_cost",
                  "form", "ep_next", "points_per_game", "status",
                  "chance_of_playing_next_round"]],
        left_on="element", right_on="id", how="left",
    )
    merged["role"] = merged.apply(
        lambda r: "Captain" if r["is_captain"] else ("Vice" if r["is_vice_captain"] else ""), axis=1
    )
    return merged, picks_data["entry_history"]
