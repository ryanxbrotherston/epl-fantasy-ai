"""
team_visuals.py — team badge URLs and colour fallback for the pitch view.

BADGE URL: `https://resources.premierleague.com/premierleague/badges/{size}/t{code}.png`,
keyed by bootstrap-static's team `code` field (NOT `id` - `id` is just this
season's 1-20 ordering and isn't stable/meaningful outside it; `code` is
the Premier League's own stable per-club identifier). Verified live
2026-08-20: fetched real badges for 6+ clubs across sizes 25/50/70,
confirmed 200 + real image/png content (downloaded and visually checked
Arsenal's, a genuine crest, not a placeholder). This is the Premier
League's own official CDN, the same one their own site and FPL's own
player-photo URLs live on (`resources.premierleague.com`).

COLOUR FALLBACK: bootstrap-static's team objects carry no colour field at
all (checked the full live schema - only code/name/short_name/strength
fields, nothing visual beyond `code`). TEAM_COLORS below is a static,
hand-maintained mapping of each club's well-known primary shirt colour,
keyed by `code` - a design decision, not derived from any API, since none
exposes it. Used only when a badge image fails to load client-side.
"""

BADGE_BASE = "https://resources.premierleague.com/premierleague/badges"


def badge_url(team_code: int, size: int = 50) -> str:
    return f"{BADGE_BASE}/{size}/t{team_code}.png"


# team code -> primary shirt colour (hex), for the 2026-27 Premier League's
# 20 clubs. Hand-maintained, not sourced from any API - see module docstring.
TEAM_COLORS: dict[int, str] = {
    3: "#EF0107",   # Arsenal
    7: "#670E36",   # Aston Villa
    91: "#DA291C",  # Bournemouth
    94: "#E30613",  # Brentford
    36: "#0057B8",  # Brighton
    8: "#034694",   # Chelsea
    9: "#78BE20",   # Coventry City
    31: "#1B458F",  # Crystal Palace
    11: "#003399",  # Everton
    54: "#000000",  # Fulham
    88: "#F5A623",  # Hull City
    40: "#0044A9",  # Ipswich Town
    2: "#1D428A",   # Leeds
    14: "#C8102E",  # Liverpool
    43: "#6CABDD",  # Man City
    1: "#DA291C",   # Man Utd
    4: "#241F20",   # Newcastle
    17: "#DD0000",  # Nott'm Forest
    6: "#132257",   # Spurs
    56: "#EB172B",  # Sunderland
}

DEFAULT_COLOR = "#666666"


def team_color(team_code: int) -> str:
    return TEAM_COLORS.get(team_code, DEFAULT_COLOR)
