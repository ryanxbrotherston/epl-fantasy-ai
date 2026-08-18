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
from squad_optimizer import load_predictions, pick_squad, pick_starting_xi, BUDGET

st.set_page_config(page_title="EPL Fantasy AI", page_icon="⚽", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


# ---------- cached data loaders ----------

@st.cache_data(ttl=3600)
def load_bootstrap():
    return fpl_api.get_bootstrap_static()


@st.cache_data(ttl=3600)
def load_base_predictions():
    """The pre-trained model's blended predictions from our seed pipeline
    (run offline via src/build_gw1_features.py + src/squad_optimizer.py)."""
    return load_predictions()


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

    from squad_optimizer import BLEND_WEIGHTS, UNAVAILABLE_STATUSES
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
    return merged


# ---------- sidebar: gameweek status ----------

st.title("⚽ EPL Fantasy AI")

try:
    bootstrap = load_bootstrap()
    gw = fpl_api.current_gameweek(bootstrap)
    st.sidebar.metric("Next deadline", gw["name"])
    st.sidebar.caption(gw["deadline_time"].replace("T", " ").replace("Z", " UTC"))
    live_ok = True
except Exception as e:
    st.sidebar.error(f"Couldn't reach the live FPL API: {e}")
    st.sidebar.caption("Falling back to the offline seed data from the last build.")
    bootstrap = None
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

        st.markdown(f"**Captain:** {captain['web_name']} ({captain['predicted_points']:.1f} pred pts)  \n"
                    f"**Vice-captain:** {vice['web_name']} ({vice['predicted_points']:.1f} pred pts)")

        st.markdown("##### Starting XI")
        st.dataframe(
            xi[["web_name", "position", "now_cost", "predicted_points"]]
            .assign(now_cost=lambda d: (d["now_cost"] / 10).map("£{:.1f}m".format))
            .rename(columns={"web_name": "Player", "position": "Pos", "now_cost": "Price", "predicted_points": "Pred pts"}),
            hide_index=True, use_container_width=True,
        )

        st.markdown("##### Bench")
        st.dataframe(
            bench[["web_name", "position", "now_cost", "predicted_points"]]
            .assign(now_cost=lambda d: (d["now_cost"] / 10).map("£{:.1f}m".format))
            .rename(columns={"web_name": "Player", "position": "Pos", "now_cost": "Price", "predicted_points": "Pred pts"}),
            hide_index=True, use_container_width=True,
        )

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
                    st.dataframe(
                        picks_df[["web_name", "position", "team_short", "role", "now_cost", "ep_next"]]
                        .assign(now_cost=lambda d: (d["now_cost"] / 10).map("£{:.1f}m".format))
                        .rename(columns={"web_name": "Player", "position": "Pos", "team_short": "Club",
                                          "role": "Role", "now_cost": "Price", "ep_next": "FPL's next-GW est."}),
                        hide_index=True, use_container_width=True,
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Bank", f"£{history['bank'] / 10:.1f}m")
                    c2.metric("Team value", f"£{history['value'] / 10:.1f}m")


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
    if entry.get("summary_overall_points"):
        st.caption(f"Overall points: {entry['summary_overall_points']} · Overall rank: {entry.get('summary_overall_rank', 'N/A'):,}")

    current_event = gw["id"] if gw["is_current"] else max(gw["id"] - 1, 1)
    result = fpl_api.team_picks_dataframe(int(team_id), current_event, bootstrap)

    if result is None:
        st.info(f"No squad submitted yet for Gameweek {current_event}, or picks aren't public until "
                f"the deadline passes. Check back after **{gw['deadline_time']}**.")
        return

    picks_df, history = result
    st.markdown(f"##### Gameweek {current_event} squad")
    st.dataframe(
        picks_df[["web_name", "position", "team_short", "role", "now_cost", "ep_next"]]
        .assign(now_cost=lambda d: (d["now_cost"] / 10).map("£{:.1f}m".format))
        .rename(columns={"web_name": "Player", "position": "Pos", "team_short": "Club",
                          "role": "Role", "now_cost": "Price", "ep_next": "FPL's next-GW est."}),
        hide_index=True, use_container_width=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Bank", f"£{history['bank'] / 10:.1f}m")
    c2.metric("Team value", f"£{history['value'] / 10:.1f}m")
    c3.metric("Free transfers used", history.get("event_transfers", 0))


with tab_mine:
    st.subheader("Your team")
    render_team_lookup("mine")


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
