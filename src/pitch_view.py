"""
pitch_view.py — reusable pitch-based squad display, rendered as raw HTML
via st.markdown(..., unsafe_allow_html=True) rather than st.dataframe().

Deliberately dataframe-agnostic: app.py's three tabs each build squads
from different sources with different columns (the propose-mode
optimizer output vs. fpl_api.team_picks_dataframe()'s real-account picks)
- rather than have this module know about either shape, callers normalize
into a plain list of "card" dicts first (see `build_card`) and this
module only ever renders cards. Keeps the one genuinely reusable piece
(the pitch/formation/card layout) decoupled from the two different data
shapes that feed it.

Card dict shape (all keys required):
  id, name, position ('GK'/'DEF'/'MID'/'FWD'), team_code (bootstrap-
  static's stable per-club `code`, NOT the season-local `id`), points
  (float), is_captain (bool), is_vice (bool), flag ('green'/'yellow'/
  'red'/'unknown'), basis (human-readable string, e.g. "confirmed team
  sheet"), detail (the full explanation string, shown on hover).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import team_visuals

FLAG_COLOR = {"green": "#2ECC71", "yellow": "#F5C518", "red": "#E63946", "unknown": "#9CA3AF"}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]


def formation_from_xi(xi_cards: list[dict]) -> str:
    """'D-M-F' string from whatever's actually in the starting XI - not
    hardcoded to any particular formation, so this handles any valid FPL
    shape (3-4-3, 3-5-2, 4-4-2, 4-3-3, 4-5-1, 5-3-2, 5-4-1, ...) as long
    as the XI itself is valid (1 GK + 10 outfield)."""
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for card in xi_cards:
        if card["position"] in counts:
            counts[card["position"]] += 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def build_card(id_, name, position, team_code, points, is_captain, is_vice, flag_result: dict) -> dict:
    """Normalizes a flag-result dict (flag/basis/detail, where `basis` is
    already the human-readable label - see app.py's _BASIS_LABEL) plus
    the rest of a player's display data into one card - the one place
    this shape gets assembled, so callers don't duplicate the field list."""
    return {
        "id": id_, "name": name, "position": position, "team_code": team_code,
        "points": points, "is_captain": is_captain, "is_vice": is_vice,
        "flag": flag_result["flag"], "basis": flag_result["basis"], "detail": flag_result["detail"],
    }


def _player_card_html(card: dict) -> str:
    badge_url = team_visuals.badge_url(card["team_code"], size=50)
    fallback_color = team_visuals.team_color(card["team_code"])
    flag_color = FLAG_COLOR.get(card["flag"], FLAG_COLOR["unknown"])
    armband = ""
    if card["is_captain"]:
        armband = '<span class="pv-armband pv-captain" title="Captain">C</span>'
    elif card["is_vice"]:
        armband = '<span class="pv-armband pv-vice" title="Vice-captain">V</span>'

    points_display = f"{card['points']:.1f}"
    points_class = "pv-points"
    if card["is_captain"]:
        points_display = f"{card['points']:.1f} (x2)"
        points_class += " pv-points-captain"

    tooltip = f"{card['basis']}: {card['detail']}"
    toggle_js = (
        "var p=this.nextElementSibling; "
        "p.style.display = (p.style.display === 'block') ? 'none' : 'block';"
    )

    return f"""
    <div class="pv-card">
        <div class="pv-badge-wrap">
            <img src="{badge_url}" class="pv-badge"
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
            <div class="pv-badge-fallback" style="display:none; background:{fallback_color};"></div>
            {armband}
            <span class="pv-flag-dot" style="background:{flag_color};" title="{tooltip}"
                  onclick="{toggle_js}"></span>
            <div class="pv-flag-popover">{tooltip}</div>
        </div>
        <div class="pv-name"><span class="pv-pos">{card['position']}</span>{card['name']}</div>
        <div class="{points_class}">{points_display}</div>
    </div>
    """


def _row_html(cards: list[dict]) -> str:
    cards_html = "".join(_player_card_html(c) for c in cards)
    return f'<div class="pv-row">{cards_html}</div>'


def render_pitch(xi_cards: list[dict], bench_cards: list[dict]) -> str:
    """Full pitch HTML: grass background, halfway line, center circle,
    penalty boxes, one row per outfield position line (sized dynamically
    from whatever formation the real XI is - see formation_from_xi),
    GK row always alone at the back, bench strip below in the same style."""
    gk = [c for c in xi_cards if c["position"] == "GK"]
    outfield_rows = []
    for pos in ["DEF", "MID", "FWD"]:
        row = [c for c in xi_cards if c["position"] == pos]
        if row:
            outfield_rows.append(row)

    rows_html = _row_html(gk) + "".join(_row_html(r) for r in outfield_rows)

    bench_html = ""
    if bench_cards:
        bench_html = f"""
        <div class="pv-bench-label">Bench</div>
        <div class="pv-bench-strip">{"".join(_player_card_html(c) for c in bench_cards)}</div>
        """

    return f"""
    <div class="pv-pitch">
        <div class="pv-halfway-line"></div>
        <div class="pv-center-circle"></div>
        <div class="pv-penalty-box pv-penalty-top"></div>
        <div class="pv-penalty-box pv-penalty-bottom"></div>
        <div class="pv-formation-rows">{rows_html}</div>
    </div>
    {bench_html}
    """


PITCH_CSS = """
<style>
.pv-pitch {
    position: relative;
    width: 100%;
    aspect-ratio: 3 / 2.4;
    max-height: 640px;
    border-radius: 12px;
    overflow: hidden;
    background: repeating-linear-gradient(
        to bottom,
        #3E9B4F 0, #3E9B4F 12.5%,
        #379046 12.5%, #379046 25%
    );
    border: 3px solid #FFFFFF55;
    margin-bottom: 0.75rem;
}
.pv-halfway-line {
    position: absolute; left: 0; right: 0; top: 50%;
    height: 2px; background: #FFFFFF88;
}
.pv-center-circle {
    position: absolute; left: 50%; top: 50%;
    width: 14%; aspect-ratio: 1;
    border: 2px solid #FFFFFF88; border-radius: 50%;
    transform: translate(-50%, -50%);
}
.pv-penalty-box {
    position: absolute; left: 30%; width: 40%; height: 14%;
    border: 2px solid #FFFFFF88; border-top: none;
}
.pv-penalty-box.pv-penalty-top { top: 0; border-top: none; border-bottom: 2px solid #FFFFFF88; border-top-width:0; }
.pv-penalty-box.pv-penalty-bottom { bottom: 0; }
.pv-formation-rows {
    position: absolute; inset: 6% 2%;
    display: flex; flex-direction: column; justify-content: space-between;
}
.pv-row {
    display: flex; justify-content: space-evenly; align-items: flex-start;
}
.pv-bench-label {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    color: #534AB7; margin: 0.5rem 0 0.25rem 0;
}
.pv-bench-strip {
    display: flex; justify-content: flex-start; gap: 1.2rem; flex-wrap: wrap;
    background: #EEEDFE; border-radius: 10px; padding: 0.75rem;
}
.pv-card {
    display: flex; flex-direction: column; align-items: center;
    width: 78px; text-align: center;
}
.pv-badge-wrap { position: relative; width: 40px; height: 40px; }
.pv-badge { width: 40px; height: 40px; object-fit: contain;
            filter: drop-shadow(0 1px 2px rgba(0,0,0,0.4)); }
.pv-badge-fallback {
    width: 40px; height: 40px; border-radius: 50%;
    align-items: center; justify-content: center;
}
.pv-armband {
    position: absolute; top: -6px; right: -6px;
    width: 18px; height: 18px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-size: 11px; font-weight: 700;
    color: white; border: 1.5px solid white;
}
.pv-captain { background: #E90052; }
.pv-vice { background: #534AB7; }
.pv-flag-dot {
    position: absolute; bottom: -2px; right: -2px;
    width: 12px; height: 12px; border-radius: 50%;
    border: 1.5px solid white;
    cursor: pointer;
}
.pv-flag-dot::after {
    content: ''; position: absolute; inset: -9px;
}
.pv-flag-popover {
    display: none;
    position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    margin-top: 6px; width: max-content; max-width: 180px;
    background: #2B2B45; color: white;
    font-family: 'Space Grotesk', sans-serif; font-size: 11px; font-weight: 500;
    line-height: 1.35; padding: 6px 8px; border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 20; text-align: left;
}
.pv-flag-popover::before {
    content: ''; position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent; border-bottom-color: #2B2B45;
}
.pv-name {
    font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 600;
    color: white; background: rgba(0,0,0,0.55); border-radius: 4px;
    padding: 1px 6px; margin-top: 4px; white-space: nowrap;
    max-width: 76px; overflow: hidden; text-overflow: ellipsis;
}
.pv-pos {
    color: #B8B5E8; font-weight: 500; margin-right: 3px;
}
.pv-points {
    font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700;
    color: #FFD400; margin-top: 2px;
}
.pv-points-captain {
    font-size: 14px; color: #FFD400; background: #E90052;
    border-radius: 4px; padding: 1px 6px;
}
.pv-bench-strip .pv-name { color: #2B2B45; background: transparent; }
.pv-bench-strip .pv-points { color: #534AB7; }
</style>
"""
