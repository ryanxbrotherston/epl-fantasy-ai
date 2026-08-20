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

from datetime import datetime, timezone

import streamlit as st
from supabase import create_client, Client

TABLE = "manager_profiles"


@st.cache_resource
def get_client() -> Client:
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["secret_key"])


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
