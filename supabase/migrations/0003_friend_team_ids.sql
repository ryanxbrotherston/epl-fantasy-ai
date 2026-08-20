-- friend_team_ids: a logged-in manager's persisted Friends list, so it
-- survives a refresh/new device instead of resetting every visit (the
-- previous st.session_state-only behavior, documented as a known gap in
-- app.py's Friends tab caption).
--
-- Same RLS pattern as manager_profiles (see 0001_manager_profiles.sql) -
-- enabled with NO policies attached, by design. Every read/write goes
-- through the Streamlit app's own backend using Supabase's SECRET key
-- (never the browser, never the publishable key), which bypasses RLS
-- entirely. Enabling RLS with zero policies means the publishable key -
-- if it were ever accidentally used against this table - gets nothing.

create table if not exists friend_team_ids (
  id              bigint generated always as identity primary key,
  google_sub      text not null references manager_profiles(google_sub),
  friend_team_id  integer not null,
  added_at        timestamptz not null default now(),
  unique (google_sub, friend_team_id)
);

alter table friend_team_ids enable row level security;
