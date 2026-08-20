"""
confirmed_lineup.py — official confirmed starting lineup, via
highlightly.net's public matches/lineups API (soccer.highlightly.net).

Verified 2026-08-20 against real live data (see NEXT_STEPS.md's
"Confirmed-lineup source found" section for the full trail): real
current-season club/player data, ToS clean for this use case, free tier
genuinely includes lineups (100 req/day, no paywall - unlike
API-Football/football-data.org, both of which gate lineups behind a paid
tier). There is no separate "predicted" mode - the endpoint returns an
empty/"Unknown"-formation lineup until one is genuinely released (per
their own docs, ~30-40 min before kickoff, once clubs confirm it), so
presence of non-empty data IS the confirmation signal itself - no
ambiguous boolean field to misread.

Two identity mismatches to work around, both handled here rather than
trusted blind:
  - TEAM NAMES differ between FPL and Highlightly for 4 clubs (Man City/
    Manchester City, Man Utd/Manchester United, Spurs/Tottenham,
    Nott'm Forest/Nottingham Forest) - TEAM_NAME_OVERRIDES below, built by
    diffing both sources' real live 20-club lists directly, not guessed.
    The other 16 clubs match via exact-string or substring comparison.
  - PLAYER IDS are entirely different schemes between the two services -
    matched by name instead (FPL's first_name+second_name vs
    Highlightly's full name, falling back to a surname/web_name
    containment check). Inherently imperfect, same caveat as
    build_gw1_features.py's existing cross-season name matching - a
    player who fails to match here should be treated as "no confirmed
    data," never as "not started."

Runs inside ai_team_monitor.py, a GitHub Actions script, NOT the
Streamlit app - HIGHLIGHTLY_API_KEY must be a GitHub Actions repository
secret (Settings -> Secrets and variables -> Actions), the same store as
AI_TEAM_ID/ALERT_EMAIL_*, not Streamlit Cloud's separate Secrets manager.
"""

import os

import requests

BASE = "https://soccer.highlightly.net"
LEAGUE_ID = 33973  # Premier League - confirmed against a live response 2026-08-20
TIMEOUT = 10

TEAM_NAME_OVERRIDES = {
    # FPL name -> Highlightly name, for the 4 clubs that don't match via
    # exact-string or substring comparison (verified 2026-08-20 by
    # diffing both sources' real live 20-club lists - see module docstring)
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
}


def _api_key() -> str | None:
    return os.environ.get("HIGHLIGHTLY_API_KEY")


def _get(path: str, params: dict | None = None):
    key = _api_key()
    if not key:
        return None
    try:
        resp = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT,
                             headers={"x-rapidapi-key": key})
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data) if isinstance(data, dict) else data
    except (requests.RequestException, ValueError):
        return None


def _normalize(name: str) -> str:
    return name.lower().strip()


def teams_match(fpl_name: str, highlightly_name: str) -> bool:
    if TEAM_NAME_OVERRIDES.get(fpl_name) == highlightly_name:
        return True
    a, b = _normalize(fpl_name), _normalize(highlightly_name)
    return a == b or a in b or b in a


def find_match_id(fpl_home: str, fpl_away: str, kickoff_iso: str) -> int | None:
    """Highlightly's own match id for a real FPL fixture, matched by team
    name + kickoff date (not exact timestamp - kickoff times occasionally
    drift by minutes between sources). Returns None if unreachable or no
    match found - both are "can't check this one," treated identically."""
    kickoff_date = kickoff_iso[:10]
    matches = _get("/matches", {"leagueId": LEAGUE_ID, "date": kickoff_date, "limit": 100})
    if matches is None:
        return None
    for m in matches:
        if teams_match(fpl_home, m["homeTeam"]["name"]) and teams_match(fpl_away, m["awayTeam"]["name"]):
            return m["id"]
    return None


def _flatten_initial_lineup(team_lineup: dict) -> set[str]:
    """initialLineup is a list of position-rows (GK row, DEF row, ...),
    each a list of player dicts - flatten to a plain set of names."""
    rows = team_lineup.get("initialLineup", [])
    return {p["name"] for row in rows for p in row}


def _flatten_substitutes(team_lineup: dict) -> set[str]:
    return {p["name"] for p in team_lineup.get("substitutes", [])}


def get_confirmed_lineup_names(highlightly_match_id: int) -> dict | None:
    """{'starters': set of names, 'bench': set of names} - home + away
    combined, since a caller checking one player only needs to know which
    bucket (if either) they landed in, not which team. None if unreachable
    or not released yet - both mean "no confirmed data right now," which
    callers must never treat as "not started.\""""
    data = _get(f"/lineups/{highlightly_match_id}")
    if data is None:
        return None
    home, away = data.get("homeTeam", {}), data.get("awayTeam", {})
    starters = _flatten_initial_lineup(home) | _flatten_initial_lineup(away)
    bench = _flatten_substitutes(home) | _flatten_substitutes(away)
    if not starters and not bench:
        return None  # not released yet
    return {"starters": starters, "bench": bench}


def player_confirmed_status(fpl_first_name: str, fpl_second_name: str, fpl_web_name: str,
                             confirmed_lineup: dict) -> bool | None:
    """True = confirmed starting. False = confirmed NOT starting (matched
    in the bench list - a genuine official downgrade, not a guess).
    None = no confident name match anywhere in this fixture's confirmed
    data - treat as unknown, never as 'not started' (see module docstring
    on name-matching limits)."""
    full_name = _normalize(f"{fpl_first_name} {fpl_second_name}")
    web = _normalize(fpl_web_name)

    def _matches_any(names: set[str]) -> bool:
        return any(_normalize(n) == full_name or web in _normalize(n) or _normalize(n).endswith(web)
                   for n in names)

    if _matches_any(confirmed_lineup["starters"]):
        return True
    if _matches_any(confirmed_lineup["bench"]):
        return False
    return None
