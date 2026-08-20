"""
app.py — EPL Fantasy AI dashboard

Three tabs:
  AI Team    - the model's optimized squad, fully automated, no human input
  My Team    - your own Team ID: pulls your real squad, flags issues
  Friends    - anyone's Team ID: same view, shareable once deployed

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import fpl_api
import lineup_predictor
import manager_store
import confirmed_lineup
import team_visuals
import pitch_view
from squad_optimizer import load_predictions, pick_squad, pick_starting_xi, BUDGET

CONFIRMED_LOOKUP_WINDOW_HOURS = 3  # a bit more generous than ai_team_monitor.py's 2h -
                                    # this is checked on-demand by page loads, not a fixed
                                    # hourly cron, so a wider net avoids missing a release
                                    # between visits; the 10min cache below bounds the cost

st.set_page_config(page_title="EPL Fantasy AI", page_icon="⚽", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

# Theme colors/font live in .streamlit/config.toml (Streamlit's own [theme]
# section) - this block only adds what config.toml can't do declaratively:
# real :hover states, and the active-tab fill. Streamlit's tab internals
# aren't a documented/stable public API - they already broke once (was
# button[data-baseweb="tab"], silently stopped matching after a version
# bump). Currently div[data-testid="stTab"] with role="tab" and
# aria-selected="true"/"false", confirmed by inspecting the deployed app's
# actual DOM. If tabs stop highlighting again after a future Streamlit
# upgrade, re-inspect the live DOM rather than guess a selector - this is
# the first place to check.
st.markdown("""
<style>
div[data-testid="stTab"] {
    transition: background-color 0.15s ease, color 0.15s ease;
    border-radius: 8px 8px 0 0;
    cursor: pointer;
}
div[data-testid="stTab"]:hover {
    background-color: #534AB7 !important;
}
div[data-testid="stTab"]:hover p {
    color: #FFFFFF !important;
}
div[data-testid="stTab"][aria-selected="true"] {
    background-color: #534AB7 !important;
}
div[data-testid="stTab"][aria-selected="true"] p {
    color: #FFFFFF !important;
}
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    border: 1px solid #534AB733;
}
</style>
""", unsafe_allow_html=True)


# ---------- cached data loaders ----------

@st.cache_data(ttl=3600)
def load_bootstrap():
    return fpl_api.get_bootstrap_static()


@st.cache_data(ttl=3600)
def load_base_predictions():
    """The pre-trained model's blended predictions from our seed pipeline
    (run offline via src/build_gw1_features.py + src/squad_optimizer.py)."""
    return load_predictions()


@st.cache_data(ttl=900)
def load_predicted_starting_ids():
    """The EARLY-prediction signal (fpledits.com's predicted lineups) -
    15min cache since this is refreshed through the week as team news
    develops, unlike the hourly-cached bootstrap/predictions above."""
    return lineup_predictor.fetch_predicted_starting_ids()


@st.cache_data(ttl=600)
def load_confirmed_lineups_for_gameweek(_bootstrap: dict, target_event: int) -> dict:
    """{team_id: {'starters': set[name], 'bench': set[name]}} for every
    club whose fixture this gameweek is within CONFIRMED_LOOKUP_WINDOW_HOURS
    of kickoff AND has a genuinely released highlightly.net lineup -
    everyone else just isn't in this dict (same "no data yet" contract as
    confirmed_lineup.py itself). 10min cache, shared across every visitor
    hitting this deployed app - keeps this bounded well under
    Highlightly's 100 req/day free tier regardless of traffic, same
    principle as the kickoff-window gating in ai_team_monitor.py.
    Leading underscore on _bootstrap tells st.cache_data not to hash it
    (it's large and already itself cached/stable within its own TTL)."""
    from datetime import datetime, timezone
    team_id_to_name = {t["id"]: t["name"] for t in _bootstrap["teams"]}
    fixtures = fpl_api.get_fixtures(target_event)
    now = datetime.now(timezone.utc)
    result = {}
    for fixture in fixtures:
        if fixture.get("kickoff_time") is None:
            continue
        kickoff = datetime.fromisoformat(fixture["kickoff_time"].replace("Z", "+00:00"))
        hours_to_kickoff = (kickoff - now).total_seconds() / 3600
        if not (-0.5 <= hours_to_kickoff <= CONFIRMED_LOOKUP_WINDOW_HOURS):
            continue
        home_name = team_id_to_name[fixture["team_h"]]
        away_name = team_id_to_name[fixture["team_a"]]
        match_id = confirmed_lineup.find_match_id(home_name, away_name, fixture["kickoff_time"])
        if match_id is None:
            continue
        confirmed = confirmed_lineup.get_confirmed_lineup_names(match_id)
        if confirmed is None:
            continue
        result[fixture["team_h"]] = confirmed
        result[fixture["team_a"]] = confirmed
    return result


_BASIS_LABEL = {
    "confirmed": "official status", "confirmed_lineup": "confirmed team sheet",
    "early": "predicted", "none": "no data",
}


def format_rank(entry: dict) -> str:
    """entry['summary_overall_rank'] - FPL's own last-known overall rank
    field (confirmed against a live response 2026-08-20). Null pre-season
    and until a manager's first gameweek is scored - not a bug, that's
    genuinely FPL's own state right now."""
    rank = entry.get("summary_overall_rank")
    return f"{rank:,}" if rank is not None else "N/A yet"


def compute_starting_likelihood(row: pd.Series, id_col: str, predicted_ids: set | None,
                                 confirmed_lineups: dict, names_by_id: pd.DataFrame) -> dict:
    """The single shared per-player flag computation - both the list-view
    'Starting?' column and the pitch view's flag indicators are built
    from this, so the two can never drift apart. Returns
    lineup_predictor.starting_likelihood_flag()'s raw dict."""
    player_id = row[id_col]
    confirmed_status = None
    team_id = row.get("team")
    confirmed = confirmed_lineups.get(team_id) if team_id is not None else None
    if confirmed is not None and player_id in names_by_id.index:
        first, second = names_by_id.loc[player_id, ["first_name", "second_name"]]
        confirmed_status = confirmed_lineup.player_confirmed_status(
            first, second, row.get("web_name", ""), confirmed
        )
    return lineup_predictor.starting_likelihood_flag(player_id, row["status"], predicted_ids, confirmed_status)


def build_pitch_cards(df: pd.DataFrame, id_col: str, points_col: str, bootstrap: dict,
                       all_players: pd.DataFrame, target_event: int,
                       captain_id=None, vice_id=None) -> list[dict]:
    """Normalizes any of the three tabs' differently-shaped squad
    dataframes into pitch_view's plain card-dict list - the one place
    that shape translation happens, so pitch_view.py itself never needs
    to know about fpl_api's column names vs. squad_optimizer's."""
    predicted_ids = load_predicted_starting_ids()
    confirmed_lineups = load_confirmed_lineups_for_gameweek(bootstrap, target_event)
    names_by_id = all_players.set_index("id")[["first_name", "second_name"]]
    team_code_by_id = {t["id"]: t["code"] for t in bootstrap["teams"]}

    cards = []
    for _, row in df.iterrows():
        result = compute_starting_likelihood(row, id_col, predicted_ids, confirmed_lineups, names_by_id)
        points = pd.to_numeric(pd.Series([row.get(points_col)]), errors="coerce").iloc[0]
        points = 0.0 if pd.isna(points) else float(points)
        cards.append(pitch_view.build_card(
            id_=row[id_col], name=row["web_name"], position=row["position"],
            team_code=team_code_by_id.get(row["team"], -1), points=points,
            is_captain=row[id_col] == captain_id, is_vice=row[id_col] == vice_id,
            flag_result={"flag": result["flag"], "basis": _BASIS_LABEL[result["basis"]], "detail": result["detail"]},
        ))
    return cards


def render_squad_pitch(xi_df: pd.DataFrame, bench_df: pd.DataFrame, id_col: str, points_col: str,
                        bootstrap: dict, all_players: pd.DataFrame, target_event: int,
                        captain_id=None, vice_id=None):
    """The one call site every tab uses to go from a squad's XI/bench
    dataframes to a rendered pitch - formation is derived from whatever's
    actually in xi_df, not assumed."""
    xi_cards = build_pitch_cards(xi_df, id_col, points_col, bootstrap, all_players, target_event,
                                  captain_id, vice_id)
    bench_cards = build_pitch_cards(bench_df, id_col, points_col, bootstrap, all_players, target_event,
                                     captain_id, vice_id)
    st.caption(f"Formation: {pitch_view.formation_from_xi(xi_cards)}")
    st.markdown(pitch_view.PITCH_CSS + pitch_view.render_pitch(xi_cards, bench_cards), unsafe_allow_html=True)


def refresh_predictions_with_live_data(base_preds: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    """Prices, injury status, and FPL's own ep_next drift daily pre-season -
    pull those fresh from bootstrap-static rather than trusting the snapshot
    the model was seeded from, while keeping our trained model's point
    estimate (that part doesn't go stale day to day)."""
    live = fpl_api.players_dataframe(bootstrap)
    live_cols = live[["id", "now_cost", "status", "ep_next", "points_per_game", "chance_of_playing_next_round"]]

    merged = base_preds.drop(
        columns=["now_cost", "status", "ep_next", "points_per_game", "chance_of_playing_next_round"],
        errors="ignore",
    ).merge(live_cols, on="id", how="left")

    from squad_optimizer import BLEND_WEIGHTS, UNAVAILABLE_STATUSES, apply_price_bias_correction
    ep_next = pd.to_numeric(merged["ep_next"], errors="coerce").fillna(merged["model_points"])
    ppg = pd.to_numeric(merged["points_per_game"], errors="coerce").fillna(merged["model_points"])
    merged["predicted_points"] = (
        BLEND_WEIGHTS["model"] * merged["model_points"]
        + BLEND_WEIGHTS["ep_next"] * ep_next
        + BLEND_WEIGHTS["ppg"] * ppg
    )
    chance = pd.to_numeric(merged["chance_of_playing_next_round"], errors="coerce")
    doubtful = merged["status"].eq("d") & chance.notna()
    merged.loc[doubtful, "predicted_points"] *= (chance[doubtful] / 100)
    merged["available"] = ~merged["status"].isin(UNAVAILABLE_STATUSES)
    merged = apply_price_bias_correction(merged)
    return merged


# ---------- sidebar: gameweek status ----------

st.title("⚽ EPL Fantasy AI")
st.caption("The dot on each player's badge = starting likelihood (hover for detail). "
           "🟢🟡🔴 with **official status** = FPL's own status field. With **confirmed team sheet** = "
           "highlightly.net's real released lineup (only exists ~30-40min pre-kickoff). With "
           "**predicted** = fpledits.com's third-party guess, refreshed through the week - not official.")

try:
    bootstrap = load_bootstrap()
    gw = fpl_api.current_gameweek(bootstrap)
    st.sidebar.metric("Next deadline", gw["name"])
    st.sidebar.caption(gw["deadline_time"].replace("T", " ").replace("Z", " UTC"))
    all_players = fpl_api.players_dataframe(bootstrap)
    live_ok = True
except Exception as e:
    st.sidebar.error(f"Couldn't reach the live FPL API: {e}")
    st.sidebar.caption("Falling back to the offline seed data from the last build.")
    bootstrap = None
    all_players = None
    live_ok = False


tab_ai, tab_mine, tab_friends = st.tabs(["🤖 AI Team", "👤 My Team", "👥 Friends"])


# ---------- AI TEAM ----------

def get_ai_team_id():
    """AI_TEAM_ID is only set once you've actually created the AI-run FPL
    account and want the site to switch from 'recommendation mode' to
    'live read-only display of the real team'. Configure it in Streamlit
    Cloud's secrets (not committed to the repo) or as an env var locally."""
    try:
        return int(st.secrets["AI_TEAM_ID"])
    except Exception:
        import os
        val = os.environ.get("AI_TEAM_ID")
        return int(val) if val else None


with tab_ai:
    st.subheader("The AI-run team")
    ai_team_id = get_ai_team_id()

    if ai_team_id is None:
        st.caption("Every decision here is made by the model. No human input. "
                   "**Not yet submitted to a real FPL account** — this is the current recommendation.")

        base = load_base_predictions()
        preds = refresh_predictions_with_live_data(base, bootstrap) if live_ok else base

        squad = pick_squad(preds, budget=BUDGET)
        xi, bench, formation, captain, vice = pick_starting_xi(squad)

        spend = squad["now_cost"].sum() / 10
        c1, c2, c3 = st.columns(3)
        c1.metric("Squad value", f"£{spend:.1f}m / £100.0m")
        c2.metric("Formation", formation)
        c3.metric("Predicted XI points", f"{xi['predicted_points'].sum():.1f}")

        if live_ok:
            render_squad_pitch(xi, bench, "id", "predicted_points", bootstrap, all_players,
                                gw["id"], captain_id=captain["id"], vice_id=vice["id"])
        else:
            st.info("Live FPL API unreachable - starting-likelihood flags need it, so the pitch "
                    "view is skipped this run. Try again shortly.")

    else:
        st.caption("🔒 Live and locked — this reads directly from the AI's real FPL account. "
                   "Nobody viewing this page, including friends, can change it. Changes only "
                   "happen when Ryan executes an alert email on the real account.")

        if not live_ok:
            st.error("Can't reach the live FPL API right now, so the AI team can't be displayed.")
        else:
            entry = fpl_api.get_entry(ai_team_id)
            if entry is None:
                st.error("AI_TEAM_ID is set but no matching FPL team was found — check the ID.")
            else:
                st.success(f"**{entry['name']}**")
                current_event = gw["id"] if gw["is_current"] else max(gw["id"] - 1, 1)
                result = fpl_api.team_picks_dataframe(ai_team_id, current_event, bootstrap)
                if result is None:
                    st.info(f"No squad locked in yet for Gameweek {current_event}.")
                else:
                    picks_df, history = result
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total points", entry.get("summary_overall_points") or "N/A yet")
                    c2.metric("Overall rank", format_rank(entry))
                    c3.metric("Bank", f"£{history['bank'] / 10:.1f}m")
                    c4.metric("Team value", f"£{history['value'] / 10:.1f}m")

                    xi = picks_df[picks_df["multiplier"] > 0]
                    bench = picks_df[picks_df["multiplier"] == 0]
                    captain_row = picks_df[picks_df["is_captain"]]
                    vice_row = picks_df[picks_df["is_vice_captain"]]
                    render_squad_pitch(
                        xi, bench, "element", "ep_next", bootstrap, all_players, current_event,
                        captain_id=captain_row["element"].iloc[0] if not captain_row.empty else None,
                        vice_id=vice_row["element"].iloc[0] if not vice_row.empty else None,
                    )


# ---------- shared: team lookup by ID ----------

def render_team_lookup(key_prefix: str, default_id: int | None = None):
    team_id = st.number_input("FPL Team ID", min_value=1, step=1, value=default_id, key=f"{key_prefix}_id",
                               help="Find this in the FPL site URL when viewing 'Pick Team' or 'Gameweek history'.")
    if not st.button("Load team", key=f"{key_prefix}_load"):
        return

    if not live_ok:
        st.error("Live FPL API isn't reachable right now, so I can't pull a real team. Try again shortly.")
        return

    entry = fpl_api.get_entry(int(team_id))
    if entry is None:
        st.error(f"No FPL team found with ID {team_id} — double check the number.")
        return

    st.success(f"**{entry['name']}** — managed by {entry['player_first_name']} {entry['player_last_name']}")

    current_event = gw["id"] if gw["is_current"] else max(gw["id"] - 1, 1)
    result = fpl_api.team_picks_dataframe(int(team_id), current_event, bootstrap)

    if result is None:
        st.info(f"No squad submitted yet for Gameweek {current_event}, or picks aren't public until "
                f"the deadline passes. Check back after **{gw['deadline_time']}**.")
        return

    picks_df, history = result
    st.markdown(f"##### Gameweek {current_event} squad")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total points", entry.get("summary_overall_points") or "N/A yet")
    c2.metric("Overall rank", format_rank(entry))
    c3.metric("Bank", f"£{history['bank'] / 10:.1f}m")
    c4.metric("Team value", f"£{history['value'] / 10:.1f}m")
    c5.metric("Free transfers used", history.get("event_transfers", 0))

    xi = picks_df[picks_df["multiplier"] > 0]
    bench = picks_df[picks_df["multiplier"] == 0]
    captain_row = picks_df[picks_df["is_captain"]]
    vice_row = picks_df[picks_df["is_vice_captain"]]
    render_squad_pitch(
        xi, bench, "element", "ep_next", bootstrap, all_players, current_event,
        captain_id=captain_row["element"].iloc[0] if not captain_row.empty else None,
        vice_id=vice_row["element"].iloc[0] if not vice_row.empty else None,
    )


with tab_mine:
    st.subheader("Your team")

    if not st.user.is_logged_in:
        st.info("Log in with Google to save your Team ID once and have it remembered on "
                 "every future visit - no re-entering it, no browser-local tricks that "
                 "break if you switch devices.")
        st.button("Log in with Google", on_click=st.login, key="mine_login")
        st.caption("Or skip login and just look your team up for this visit:")
        render_team_lookup("mine_guest")
    else:
        top_l, top_r = st.columns([4, 1])
        top_l.caption(f"Logged in as {st.user.email}")
        if top_r.button("Log out", key="mine_logout"):
            st.logout()

        try:
            profile = manager_store.get_manager_profile(st.user.sub)
            store_error = None
        except Exception as e:
            profile, store_error = None, e

        if store_error is not None:
            st.error(f"Couldn't reach the persistent store right now: {store_error}")
            st.caption("Falling back to one-off entry - it just won't be remembered this time.")
            render_team_lookup("mine")
        else:
            saved_team_id = profile.get("fpl_team_id") if profile else None

            if saved_team_id is None:
                st.markdown("**First time here** — enter your FPL Team ID once and it'll be "
                            "remembered for next time.")
                new_id = st.number_input(
                    "Your FPL Team ID", min_value=1, step=1, key="mine_first_id",
                    help="Find this in the FPL site URL when viewing 'Pick Team' or 'Gameweek history'.",
                )
                if st.button("Save and load my team", key="mine_save_first"):
                    manager_store.save_fpl_team_id(st.user.sub, st.user.email, st.user.name, int(new_id))
                    st.rerun()
            else:
                st.success(f"Team ID **{saved_team_id}** remembered — no need to re-enter it.")

                def _on_toggle_email_alerts():
                    manager_store.save_email_alerts_enabled(st.user.sub, st.session_state.mine_email_alerts)

                st.checkbox(
                    "Email me if my starting XI has an injury/suspension issue",
                    value=profile.get("email_alerts_enabled", False),
                    key="mine_email_alerts",
                    on_change=_on_toggle_email_alerts,
                    help="Opt-in only - nobody gets emailed without checking this themselves, "
                         "and unchecking it stops future alerts immediately.",
                )
                with st.expander("Change my saved Team ID"):
                    updated_id = st.number_input(
                        "New FPL Team ID", min_value=1, step=1, value=saved_team_id, key="mine_update_id",
                    )
                    if st.button("Update saved Team ID", key="mine_save_update"):
                        manager_store.save_fpl_team_id(st.user.sub, st.user.email, st.user.name, int(updated_id))
                        st.rerun()
                render_team_lookup("mine", default_id=saved_team_id)


with tab_friends:
    st.subheader("Friends")
    st.caption("Anyone can enter their FPL Team ID here — no login needed, this is public FPL data.")

    if "friend_ids" not in st.session_state:
        st.session_state.friend_ids = []

    new_id = st.number_input("Add a friend's Team ID", min_value=1, step=1, key="new_friend_id")
    if st.button("Add friend"):
        if new_id not in st.session_state.friend_ids:
            st.session_state.friend_ids.append(int(new_id))

    if not st.session_state.friend_ids:
        st.info("No friends added yet this session.")
    else:
        friend_tabs = st.tabs([f"ID {fid}" for fid in st.session_state.friend_ids])
        for fid, ftab in zip(st.session_state.friend_ids, friend_tabs):
            with ftab:
                render_team_lookup(f"friend_{fid}", default_id=fid)

    st.caption("Note: this list resets when the page is refreshed - persistent storage "
               "across visits (so everyone's teams stay saved) is the next build step.")
