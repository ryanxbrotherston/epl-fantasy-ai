-- manager_profiles: persistent manager identity keyed by st.user's stable
-- Google OIDC subject id, so an FPL Team ID only needs entering once.
-- See DECISION_manager_identity.md (Option A) and NEXT_STEPS.md.
--
-- RLS is enabled with NO policies attached - by design, not an oversight.
-- Every read/write to this table goes through the Streamlit app's own
-- backend using Supabase's SECRET key (never the browser, never the
-- publishable key), which bypasses RLS entirely. Enabling RLS with zero
-- policies means the publishable key - if it were ever accidentally used
-- against this table - gets nothing, not an open table. Defense in depth
-- for an architecture that doesn't otherwise need RLS policies at all.

create table if not exists manager_profiles (
  google_sub    text primary key,
  email         text,
  display_name  text,
  fpl_team_id   integer,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

alter table manager_profiles enable row level security;
