"""
lineup_predictor.py — starting-likelihood flag for FPL squad players.

Two distinct signals, deliberately not collapsed into one number (see
NEXT_STEPS.md's "Predicted lineup integration" section for the full
research trail behind this):

  - "confirmed": FPL's own official status field (bootstrap-static's
    `status`/`chance_of_playing_next_round`) - i/s/u/n are FPL's own real
    designations (injured/suspended/unavailable/left club), not a
    prediction. ai_team_monitor.py already checked this before this
    module existed (flag_problem_players).
  - "early": fpledits.com's public predicted-lineup snapshot
    (https://fpledits.com/api/predicted-lineup/1) - a third-party,
    editorially-curated "who do we think will start" call, refreshed
    through the week as team news develops (their own `version` field
    increments on each revision, confirmed non-trivial - e.g. Arsenal's
    was at version 9 as of 2026-08-18). This is a prediction, not an
    official source - always label it as such wherever it's shown.

Deliberately does NOT do official ~60-75-min-pre-kickoff team-sheet
confirmation - every source checked either violated its ToS, was
bot-protected, had no discoverable API, or required a paid tier. See
NEXT_STEPS.md. Official confirmation stays manual (FPL's own app/site)
until a real source turns up.

fpledits.com's ToS (checked 2026-08-20, /terms) only prohibits misuse,
disruption, or unauthorized access - no restriction found on reading this
public, unauthenticated endpoint. Their /api/predicted-lineup/confirmed/...
variant requires auth (verified: returns 401) - a paid feature,
deliberately NOT used here; only the free predicted-lineup endpoint.

KNOWN DATA QUIRK (verified 2026-08-20): the endpoint's own teamId/teamName
fields are stale - they reference an old season's 20-club list (e.g.
"Burnley" appears where the real current club in that slot is
Bournemouth). The player-level data itself is NOT stale: every player id
in `selectedLineup` was cross-checked against live bootstrap-static and
correctly resolves to that player's real current team. This module
deliberately ignores teamId/teamName entirely and joins every player by
FPL element id instead, sidestepping the bug rather than trusting the
(broken) team labels.
"""

import requests

FPLEDITS_PREDICTED_LINEUP_URL = "https://fpledits.com/api/predicted-lineup/1"
TIMEOUT = 10
UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}
USER_AGENT = ("epl-fantasy-ai-personal-use/1.0 "
              "(+https://github.com/ryanxbrotherston/epl-fantasy-ai)")


def fetch_predicted_starting_ids() -> set[int] | None:
    """Every player id currently in ANY club's predicted starting XI, per
    fpledits.com's latest snapshot (ids only - team labels ignored, see
    module docstring). Returns None (not an empty set) on any fetch/parse
    failure, so callers can tell "couldn't reach the source" apart from
    "nobody's predicted to start" (which can't genuinely happen) - a
    fetch failure must never get silently treated as "everyone's benched"."""
    try:
        resp = requests.get(FPLEDITS_PREDICTED_LINEUP_URL, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        starters = {
            player["id"]
            for team_entry in data
            for player in team_entry.get("selectedLineup", [])
            if "id" in player
        }
        return starters or None
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None


def starting_likelihood_flag(player_id: int, status: str,
                              predicted_starting_ids: set[int] | None) -> dict:
    """One player's starting-likelihood flag, from whichever signals are
    available. Always returns {'flag': 'green'/'yellow'/'red'/'unknown',
    'basis': 'confirmed'/'early'/'none', 'detail': str} so callers can
    label the source honestly rather than presenting a guess as fact."""
    if status in UNAVAILABLE_STATUSES:
        return {"flag": "red", "basis": "confirmed",
                "detail": f"FPL status '{status}' (official)"}

    if predicted_starting_ids is None:
        if status == "d":
            return {"flag": "yellow", "basis": "confirmed",
                     "detail": "FPL status 'doubtful' (official)"}
        return {"flag": "unknown", "basis": "none",
                "detail": "early-prediction source unavailable this check"}

    in_predicted_xi = player_id in predicted_starting_ids
    if in_predicted_xi and status == "d":
        return {"flag": "yellow", "basis": "early",
                "detail": "in fpledits.com's predicted XI, but FPL lists them doubtful"}
    if in_predicted_xi:
        return {"flag": "green", "basis": "early",
                "detail": "in fpledits.com's predicted starting XI"}
    return {"flag": "red", "basis": "early",
            "detail": "not in fpledits.com's predicted starting XI"}
