-- email_alerts_enabled: per-manager opt-in for friend_alert_monitor.py's
-- injury/suspension/confirmed-lineup email alerts about their own squad.
-- See DECISION_friend_alerts.md for why this is a dedicated consent flag
-- rather than implied by having a saved fpl_team_id.
--
-- Defaults false - saving a Team ID (existing behavior, unrelated to this
-- flag) must never itself opt someone into emails they didn't ask for.
-- Unchecking it in the app is a complete unsubscribe: friend_alert_monitor.py
-- filters on this flag every run, so no separate removal step is needed.

alter table manager_profiles
  add column if not exists email_alerts_enabled boolean not null default false;
