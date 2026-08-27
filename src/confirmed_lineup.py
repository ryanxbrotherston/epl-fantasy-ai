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
from datetime import datetime

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
    """GitHub Actions secret (ai_team_monitor.py's context) OR Streamlit
    Cloud secret (app.py's context) - these are two separate secret
    stores, same distinction as AI_TEAM_ID/ALERT_EMAIL_* - so both paths
    are checked rather than assuming one. st.secrets access is wrapped
    since it raises outside an actual Streamlit runtime (e.g. when this
    module is imported from the plain ai_team_monitor.py script)."""
    key = os.environ.get("HIGHLIGHTLY_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("highlightly", {}).get("api_key")
    except Exception:
        return None


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


def find_player_id(fpl_first_name: str, fpl_second_name: str, fpl_web_name: str) -> int | None:
    """Highlightly's own player id for a real FPL player, matched by name -
    same imperfect-but-treated-as-'no data' philosophy as player_confirmed_status()
    below (different ID schemes, no shared key). Tries the full name first
    (most precise), then falls back to web_name (surname) since Highlightly's
    own /players?name= search does its own fuzzy matching server-side and a
    bare surname often succeeds where a full-name string doesn't."""
    full_name = f"{fpl_first_name} {fpl_second_name}"
    full_norm, web_norm = _normalize(full_name), _normalize(fpl_web_name)
    for query in (full_name, fpl_web_name):
        results = _get("/players", {"name": query, "limit": 10})
        if not results:
            continue
        for r in results:
            name_norm = _normalize(r.get("fullName") or r.get("name") or "")
            if name_norm == full_norm or web_norm in name_norm or name_norm.endswith(web_norm):
                return r["id"]
    return None


def get_player_injury_news(highlightly_player_id: int) -> dict | None:
    """{'injuries': [...], 'related_news': [...]} straight from Highlightly's
    own licensed player-detail endpoint - real dated injury records (reason,
    fromDate, toDate, missedGames) and real article title/url/date entries,
    NOT scraped from any news site directly (BBC's robots.txt explicitly
    forbids scraping/data-mining their content, and premierleague.com's own
    Terms of Use forbid reproducing or "creating a database" from theirs -
    see conversation 2026-08-27 for the full trail on why this app only
    ever sources news/injury data through a licensed API, never a scraper).
    None if unreachable - callers should treat that as "no data available
    right now," never as "no injury/no news.\""""
    data = _get(f"/players/{highlightly_player_id}")
    if data is None:
        return None
    return {
        "injuries": data.get("injuries") or [],
        "related_news": data.get("relatedNews") or [],
    }


def _parse_injury_date(date_str: str | None):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d.%m.%Y")
    except ValueError:
        return None


def current_injury(injuries: list[dict]) -> dict | None:
    """The most recent entry in a player's injuries array (by fromDate) - a
    reasonable proxy for "the injury behind their current status", since
    Highlightly's own docs don't distinguish past-vs-ongoing with a boolean
    (an ongoing injury's toDate is just left blank/future). FPL's own status
    field is still the source of truth for WHETHER they're currently out;
    this only supplies the extra reason/date detail once we already know
    they are. None if the list is empty or every date fails to parse."""
    dated = [(i, _parse_injury_date(i.get("fromDate"))) for i in injuries]
    dated = [(i, d) for i, d in dated if d is not None]
    if not dated:
        return None
    return max(dated, key=lambda pair: pair[1])[0]


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
