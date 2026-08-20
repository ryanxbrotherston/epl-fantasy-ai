# Decision: opt-in email alerts for friends' own squads

**Status: built.** Unlike `DECISION_manager_identity.md`, this wasn't an open
research question with options to weigh - Ryan gave an explicit design
(opt-in checkbox, reuse the AI monitor's checking logic, separate log/script)
as part of the ask itself. This doc records why that design is right, for
the same reason `DECISION_manager_identity.md` records the identity
decision: so the reasoning is traceable later, not just the code.

## What's being added

`ai_team_monitor.py` only ever emails Ryan, about the one AI-run team. The
ask: let any logged-in friend with a saved Team ID (via `manager_store.py`,
see `DECISION_manager_identity.md`) opt in to getting the same kind of
injury/suspension/confirmed-lineup email alert, but about *their own* squad,
sent to *their own* saved email.

## Why an opt-in checkbox, not automatic for everyone with a saved Team ID

Saving a Team ID (existing behavior) and consenting to be emailed are two
different acts, and conflating them would mean someone gets emails they
never asked for the moment they save a Team ID to look up their squad once.
The checkbox is the actual consent record - `email_alerts_enabled` is read
fresh every run by `friend_alert_monitor.py`, so:

- Nobody gets emailed without having explicitly checked the box themselves.
- Unchecking it is a complete, self-service unsubscribe with no extra work
  on Ryan's end - the very next run just skips them, and the dedup log entry
  for their `team_id` simply stops growing. No "please remove me" request
  needed, no manual list to maintain.

**How to apply**: any future feature that reads or writes
`email_alerts_enabled` must preserve this - never default it to `true`,
never flip it on behalf of a user as a side effect of some other action
(e.g. saving/changing a Team ID should never silently re-enable it).

## Why the shared-checks refactor (`src/squad_alert_checks.py`)

The ask was explicit: reuse `ai_team_monitor.py`'s existing injury/
suspension/confirmed-lineup logic rather than duplicating it. Two monitors
running near-identical FPL-status/confirmed-lineup/bench-likely checks
against two different squads is exactly the kind of logic that drifts apart
silently if copy-pasted (one gets a bugfix, the other doesn't). Extracting
`flag_problem_players`, `flag_confirmed_benched_players`,
`flag_bench_likely_players`, and `suggest_replacement` into
`src/squad_alert_checks.py` means both `ai_team_monitor.py` and
`friend_alert_monitor.py` call the identical functions - a fix to one
benefits both by construction. `ai_team_monitor.py`'s own behavior is
unchanged; it's just thinner now, calling into the shared module.

## Why a separate dedup log (`data/friend_alert_log.json`), not shared with `alert_log.json`

`data/alert_log.json` is keyed by gameweek → list of issue keys, implicitly
scoped to the one AI team (there's only ever one team being checked, so no
team identifier is needed in the key). Reusing that same file/format for N
friends' squads would require either:

- Changing its schema to nest by team ID (a live-format change to a file
  `ai_monitor.yml` already reads/writes every hour, risking a subtle
  breakage of the AI team's own alerting - explicitly out of scope; the ask
  was "do not touch alert_log.json's format or the AI team's existing alert
  path at all"), or
- Mixing the AI team's issue keys and friends' issue keys in one flat
  structure, where an element ID could collide across two different
  people's squads and either suppress a real alert for one of them or never
  fire it at all.

A new file, `data/friend_alert_log.json`, keyed by
`{gameweek: {team_id: [issue_keys]}}`, avoids both problems: it's
structurally suited to "many teams, one gameweek" from the start, and it
can't ever touch or corrupt the AI team's own dedup state. The two monitors'
failure domains stay fully isolated - a bug in the friend monitor's log
handling cannot break Ryan's own alerts, and vice versa.

## Why a separate script (`src/friend_alert_monitor.py`), not one script handling both

Same isolation argument as the log file, one level up: the AI team monitor
is a working, already-relied-upon path (`ai_monitor.yml`, hourly, Ryan's
only alerting mechanism for the AI-run team). Folding friend-alert looping
into the same script - looping over `manager_profiles`, running per-person
checks, sending per-person emails - adds real complexity (a Supabase read,
an N-person loop, per-recipient email addressing) to a script that
currently has none of that, for no benefit: the two jobs don't share a
schedule requirement, a failure mode, or a recipient, they only share the
*checking logic*, which is exactly what `squad_alert_checks.py` factors out
without requiring the scripts themselves to merge. A bug or an outage in
`friend_alert_monitor.py` (e.g. a bad row in `manager_profiles`, a Supabase
hiccup) therefore can't take down the AI team's own alert run, which stays
exactly as it was before this feature existed.

## Why `friend_alert_monitor.py` needs its own Supabase client path

`manager_store.get_client()` reads `st.secrets["supabase"]["url"/"secret_key"]`,
which only exists inside a running Streamlit process - `friend_alert_monitor.py`
runs standalone in GitHub Actions, same as `ai_team_monitor.py`, with no
Streamlit runtime at all. Rather than fork a second copy of the client
construction logic into the new script, `manager_store.get_client()` gained
an environment-variable fallback (`SUPABASE_URL`/`SUPABASE_SECRET_KEY`) for
when `st.secrets` isn't available, so both callers still share the one
function. Streamlit Cloud's secrets and the GitHub Actions secrets end up
holding the same values, just delivered through the mechanism each runtime
actually has access to.

## Manual step still needed

Same pattern as `HIGHLIGHTLY_API_KEY` previously: `SUPABASE_URL` and
`SUPABASE_SECRET_KEY` need adding to the repo's GitHub Actions secrets
(Settings → Secrets and variables → Actions) using the same values already
in Streamlit Cloud's secrets manager. This can't be done from here - it's a
manual step for Ryan.
