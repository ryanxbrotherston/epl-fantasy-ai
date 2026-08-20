"""
manager_store.py — persistent manager identity, keyed by st.user's stable
Google OIDC subject id (`st.user.sub`), backed by Supabase.

See DECISION_manager_identity.md (Option A, the one that shipped) and
NEXT_STEPS.md for the full research trail behind this design.

Uses Supabase's SECRET key deliberately, not the publishable key - this
module only ever runs inside app.py's own server-side Python process on
Streamlit Cloud (st.secrets never reaches the browser), so it's a trusted
backend context. The secret key bypasses Row Level Security by design;
manager_profiles' RLS is enabled with zero policies specifically so the
publishable key - if it were ever used here by mistake - gets nothing.
"""

import os
from datetime import datetime, timezone

import streamlit as st
from supabase import create_client, Client

TABLE = "manager_profiles"


@st.cache_resource
def get_client() -> Client:
    """st.secrets is only populated inside a running Streamlit process -
    friend_alert_monitor.py (see DECISION_friend_alerts.md) runs standalone
    in GitHub Actions, same as ai_team_monitor.py, with no Streamlit runtime
    at all. Falling back to plain env vars there lets both callers share
    this one function instead of friend_alert_monitor.py forking its own
    client construction."""
    try:
        url, key = st.secrets["supabase"]["url"], st.secrets["supabase"]["secret_key"]
    except (FileNotFoundError, KeyError):
        url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    return create_client(url, key)


def get_manager_profile(google_sub: str) -> dict | None:
    """This manager's saved profile (FPL Team ID, etc.), or None if
    they've never saved one before."""
    client = get_client()
    resp = client.table(TABLE).select("*").eq("google_sub", google_sub).limit(1).execute()
    return resp.data[0] if resp.data else None


def save_fpl_team_id(google_sub: str, email: str | None, display_name: str | None, team_id: int) -> None:
    """Upsert - first save creates the row, later saves (e.g. changing
    Team ID) just update it. updated_at is set explicitly rather than
    relying on a DB trigger, since Supabase's default schema has none."""
    client = get_client()
    client.table(TABLE).upsert({
        "google_sub": google_sub,
        "email": email,
        "display_name": display_name,
        "fpl_team_id": team_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def save_email_alerts_enabled(google_sub: str, enabled: bool) -> None:
    """Upsert the opt-in flag friend_alert_monitor.py filters on every run
    (see DECISION_friend_alerts.md) - this is the actual consent record, so
    it only ever changes when the manager themselves toggles it, never as a
    side effect of saving a Team ID."""
    client = get_client()
    client.table(TABLE).upsert({
        "google_sub": google_sub,
        "email_alerts_enabled": enabled,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


FRIEND_TABLE = "friend_team_ids"


def get_friend_team_ids(google_sub: str) -> list[int]:
    """This manager's persisted Friends list - empty if they've never added
    one. Ordered by when they were added, oldest first, so the tab order
    stays stable across visits instead of shuffling."""
    client = get_client()
    resp = client.table(FRIEND_TABLE).select("friend_team_id").eq("google_sub", google_sub) \
        .order("added_at").execute()
    return [row["friend_team_id"] for row in resp.data]


def add_friend_team_id(google_sub: str, team_id: int) -> None:
    """Plain insert, not an upsert - the (google_sub, friend_team_id) unique
    constraint on the table itself rejects a duplicate add (raises), so
    callers that already de-dupe against get_friend_team_ids() before
    calling this (as app.py's Friends tab does) won't normally hit it."""
    client = get_client()
    client.table(FRIEND_TABLE).insert({
        "google_sub": google_sub,
        "friend_team_id": team_id,
    }).execute()


def remove_friend_team_id(google_sub: str, team_id: int) -> None:
    client = get_client()
    client.table(FRIEND_TABLE).delete().eq("google_sub", google_sub).eq("friend_team_id", team_id).execute()
