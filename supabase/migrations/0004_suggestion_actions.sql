-- suggestion_actions: a logged-in manager's own record of what they actually
-- did about each Suggested Changes alert, per gameweek - "I made this change"
-- or "I'm skipping this one" - rather than the app silently assuming every
-- suggestion was (or wasn't) followed. See conversation 2026-08-27: "need to
-- be able to fill out what you actually ended up doing... rather than the
-- model just assuming you followed its advice."
--
-- issue_key mirrors squad_alert_checks.py's own dedup key convention
-- ({element}_{status}) so it's at least consistent in spirit with
-- ai_team_monitor.py's alert_log.json, though this is a separate,
-- user-driven log, not the email dedup log itself.
--
-- Same RLS pattern as manager_profiles/friend_team_ids (see those files) -
-- enabled with NO policies attached, by design. Every read/write goes
-- through the Streamlit app's own backend using Supabase's SECRET key
-- (never the browser, never the publishable key), which bypasses RLS
-- entirely.

create table if not exists suggestion_actions (
  google_sub  text not null references manager_profiles(google_sub),
  gameweek    integer not null,
  issue_key   text not null,
  action      text not null check (action in ('done', 'skipped')),
  updated_at  timestamptz not null default now(),
  primary key (google_sub, gameweek, issue_key)
);

alter table suggestion_actions enable row level security;
