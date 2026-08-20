# Decision: persistent manager identity + multi-user support

**Status: research only, nothing built.** Per standing instructions, this is a
genuine architecture decision (who a user "is" to the app, and where their
data lives), not a small implementation detail — Ryan picks the option, then
it gets built as a separate step.

## What's being decided

Right now, "My Team"/"Friends" require re-entering an FPL Team ID every
visit, and the Friends list is `st.session_state`-only (resets on refresh,
documented as a known gap in `app.py`). The ask: a Team ID should only need
entering once, with history tracked over time, and this needs to work for
multiple real friends, not just Ryan - "properly, with real login," not a
browser-local trick pretending to be identity.

## Researched: does the current deployment even support `st.login()`?

**Streamlit version**: `st.login()`/`st.user` shipped in **Streamlit
1.42** (Feb 2026 release). `requirements.txt` currently pins
`streamlit>=1.38` with no upper bound - locally, `pip show streamlit`
resolves to 1.60.0, well past 1.42. I could not check the exact version
actually running on the live Streamlit Community Cloud deployment (no
browser/dashboard access this session - see the standing note on not
asserting what can't be verified). Since Community Cloud rebuilds from
`requirements.txt` on each deploy with no version ceiling set, it's very
likely already on something ≥1.42, but **the honest fix regardless of what's
currently deployed is to bump the pin to `streamlit>=1.42` explicitly**, so
this stops being a "probably" - a one-line change, not something to guess
about.

**Extra dependency**: `st.login()` also requires `Authlib>=1.3.2`, which is
**not currently in `requirements.txt`** at all. This needs adding regardless
of which identity option gets picked, if login is involved.

**secrets.toml requirements**: an `[auth]` table needs `redirect_uri` and
`cookie_secret` (shared across providers), plus per-provider `client_id`,
`client_secret`, `server_metadata_url`. For Google specifically,
`server_metadata_url` is Google's fixed well-known endpoint
(`https://accounts.google.com/.well-known/openid-configuration`), so that
part needs no separate Google setup beyond getting the client ID/secret.
Community Cloud already has a Secrets manager in the app's own settings
(the same place `AI_TEAM_ID` already lives) - no new mechanism needed to
store this.

**Redirect URI on Community Cloud**: works the same way as local dev, just
pointed at the deployed URL - `https://<your-app>.streamlit.app/oauth2callback`.
Confirmed via Streamlit's own Google OIDC tutorial and multiple independent
working examples; nothing Community-Cloud-specific breaks this. One doc note
worth flagging: Streamlit's own docs mention hosted *code* environments like
GitHub Codespaces can break the login redirect due to their proxying -
Community Cloud isn't in that category (it serves the app directly at its
own domain, not through a dev-environment proxy), but it's the kind of thing
worth a real click-through test once built, not assumed clean.

Sources: [st.user](https://docs.streamlit.io/develop/api-reference/user/st.user), [st.login](https://docs.streamlit.io/develop/api-reference/user/st.login), [Authentication concepts](https://docs.streamlit.io/develop/concepts/connections/authentication), [Google OIDC tutorial](https://docs.streamlit.io/develop/tutorials/authentication/google), [Version 1.42.0 announcement](https://discuss.streamlit.io/t/version-1-42-0/92243)

## Researched: does Streamlit Community Cloud's filesystem survive reboots?

**I could not empirically test this myself** - it would require either
triggering a reboot/redeploy of the live app (needs Community Cloud
dashboard access, which this session doesn't have) or a round trip through
Ryan watching the live app before/after. What I have instead is strong,
convergent secondhand evidence, which I'm distinguishing explicitly from a
verified fact:

- A Streamlit **staff member** (Caroline), answering a user who lost a
  SQLite database and video files after a routine reboot, on the official
  forum: *"Best practice in this case if you need to access the data again
  in the future would be to use a database."* That's about as close to an
  authoritative answer as exists short of the docs page stating it outright
  (the docs page on managing apps documents "sleep after 12h inactivity"
  and a "Reboot app" control, but doesn't itself spell out filesystem
  behavior in so many words).
- Multiple independent user reports of the same symptom (local files/SQLite
  gone after reboot) in the same and related threads.
- This is also just standard behavior for free-tier container PaaS
  platforms generally (Streamlit Cloud, Render's free tier, Heroku's old
  free dynos, etc. all work this way) - not an exotic claim.

**Practical consequence either way**: this repo has already run into the
exact same failure mode twice, independently, on two different platforms -
`ai_team_monitor.py`'s `data/alert_log.json` gets committed back to *git*
specifically because GitHub Actions runners are fresh containers every run,
and Supabase's own free tier (see below) auto-pauses inactive projects the
same way the Streamlit app itself was just found asleep. **If Ryan wants
100% certainty specific to this exact deployment**, the cheap test is:
temporarily add a few lines to `app.py` that write a timestamp to a local
file on each run and display "last seen: {timestamp} (written {now})", push,
load the app once, use Community Cloud's "Reboot app" button, reload, and
see if the timestamp reset. I didn't add this speculatively since it's a
throwaway diagnostic, not real app code - happy to add it if Ryan wants the
empirical answer instead of the (already quite strong) secondhand one.

**Bottom line for the decision**: assume ephemeral, size the architecture
around it. If either Option A or C below gets picked, any persistent data
goes in an external store (Supabase), never local disk.

## Option A: `st.login()` with Google + Supabase (matches what was asked for as primary)

**How it works**: friends click "Log in with Google," Streamlit handles the
OIDC redirect, `st.user.email`/`st.user.sub` (a stable per-user ID) becomes
the key for a small Supabase table storing their FPL Team ID and whatever
history gets tracked over time. Login only needs to happen once per browser
(Streamlit's identity cookie lasts 30 days, non-configurable) - the Team ID
itself gets remembered forever server-side, not just for the cookie's life.

**Setup complexity**: moderate, one-time. Concretely:
1. Google Cloud Console: create a project (free, no billing account
   required for basic `openid`/`email`/`profile` scopes - billing is only
   needed for paid APIs like Maps/BigQuery, not plain sign-in), configure
   the OAuth consent screen, create a Web Application OAuth client, add the
   deployed app's redirect URI. ~15-30 minutes.
2. Add `client_id`/`client_secret`/`redirect_uri`/`cookie_secret` to
   Community Cloud's Secrets manager (same place `AI_TEAM_ID` already
   lives).
3. Bump `requirements.txt` (`streamlit>=1.42`, add `authlib>=1.3.2`,
   `supabase` client library).
4. Create a free Supabase project, one small table (`manager_id`,
   `google_sub`, `team_id`, plus whatever history columns), wire it in via
   `st.connection` or the `supabase-py` client.
5. Code changes: a login gate, replace session-only Team ID entry with a
   lookup/save against Supabase keyed by `st.user`'s stable ID.

**Do friends need a Google account?** Yes, explicitly - `st.login()` only
supports OIDC providers (Google, Microsoft, Okta, Auth0 per Streamlit's
docs); there's no email/password option. In practice this is a low bar
(nearly everyone has one) but it is a real requirement worth stating
plainly rather than assuming it's a non-issue.

**The 100-user Google verification question** (researched specifically,
since it changes the real complexity here a lot): Google requires formal
app verification only past 100 users, or for sensitive/restricted scopes.
For **personal-use apps under 100 users** requesting only basic scopes
(which is exactly this case - a handful of friends, `openid`/`email`/
`profile` only), Google's own help docs state verification is **not
required**: *"If the app is for your personal use (fewer than 100 users),
you and your limited number of users can continue using the app without
going through verification."* The catch: staying in "Testing" publish
status means each friend's email has to be manually added as a test user in
the Google Cloud Console first, and they'll see an "unverified app" warning
screen they have to click through ("Advanced > Go to [app] (unsafe)") -
not a hard block, just a mildly scary extra click the first time. This is
manageable for a friends-and-family scale app; would become real work again
only if it ever grew past ~100 users.

**Ongoing maintenance/cost**: **$0** for Google's side (OAuth client
credentials themselves are free; no billing account needed for these
scopes). Supabase free tier: 500MB database, unlimited API requests, 50k
MAU cap - all wildly more than this app needs. **One real gotcha,
directly analogous to what just happened with the Streamlit app itself**:
Supabase free-tier projects auto-pause after **7 days with zero API
requests** (data is retained, ~30 second wake-up on the next request) - if
the app itself also goes quiet for a week, both could be asleep
simultaneously and the first visitor eats two wake-up delays back to back.
Not a blocker, just worth knowing going in, since it's literally the same
failure mode Ryan just hit with Streamlit's own sleep behavior.

Sources: [User auth concepts](https://docs.streamlit.io/develop/concepts/connections/authentication), [Google OIDC tutorial](https://docs.streamlit.io/develop/tutorials/authentication/google), [Google: when verification is not needed](https://support.google.com/cloud/answer/13464323?hl=en), [Google: unverified apps / 100-user cap](https://support.google.com/cloud/answer/7454865?hl=en), [Supabase free tier limits 2026](https://www.itpathsolutions.com/supabase-free-tier-limits)

## Option B: browser-side persistence only (no real login) - for comparison, as requested

Store the Team ID in the browser itself (e.g. `extra-streamlit-components`'
`CookieManager`, or a localStorage-backed component) so it survives a page
refresh without re-entering it, no Google/Supabase/login flow at all.

**Setup complexity**: low - one extra package, a handful of lines to
read/write a cookie around the existing Team ID input.

**Honestly, per the explicit ask to say so**: this does **not** solve "who
is this person." It solves "remember a number on this one browser, this one
device." It doesn't give real per-person identity, doesn't support the
"history tracked over time" ask in any centralized way (any history would
also have to live browser-side, or be re-fetched fresh from FPL's own API
each time, which is fine for *this season's* history but doesn't accumulate
anything beyond what FPL's own API already returns), breaks the moment
someone clears cookies/switches browsers/switches devices, and does nothing
for "multiple friends" beyond each of them independently getting their own
forgetful-free browser. It's the "browser-local trick" the ask specifically
said not to fake this with - included here only because it was asked for as
a comparison point, not as a real contender given what was actually
requested. Also worth noting: the community threads researched here flag
real technical gotchas with these cookie components too (iframe storage
isolation between the component and the main app context has tripped people
up) - it's not even perfectly reliable at the one thing it tries to do.

**Ongoing maintenance/cost**: $0, but also delivers the least.

## Option C: Team-ID-keyed persistence, no login (a real middle option, not asked for by name but genuinely relevant)

Worth surfacing because it's a different trade-off than either A or B, not
just a cheaper version of A. Skip OAuth entirely; use the FPL Team ID
itself as the persistence key in a small Supabase table (same store as
Option A, same $0/free-tier setup), and use a browser cookie/localStorage
only to remember *which* Team ID this browser belongs to so it doesn't need
re-entering - but the actual history data lives server-side, keyed by Team
ID, so it survives cookie clears/device switches for anyone who re-enters
their own ID once.

**Setup complexity**: lower than Option A - no Google Cloud Console step,
no OIDC flow, no `Authlib` dependency, no per-user Google account
requirement. Just the Supabase table + a cookie for convenience.

**Do friends need a Google account?** No - this is the real advantage over
A.

**The honest trade-off, stated plainly**: this is *not* real access
control. FPL Team IDs are not secret - they're public numbers, visible in
any mini-league and in the URL of anyone's public team page, which is
exactly why "My Team"/"Friends" can already look anyone up with no login
today. Anyone who knows (or guesses/enumerates) a Team ID could see, and
under this option *write*, whatever's stored against it - there's no proof
that the person entering a Team ID is actually its owner. For this app's
actual stakes (fantasy football points history, nothing financial or
private), that may well be an acceptable trade-off - but it's a real one,
not a hidden one, and it's the reason the ask leaned toward "real login" in
the first place. Mentioning this option because it directly serves "a Team
ID only needs entering once, with history tracked over time" without the
Google/OIDC complexity - just flagging that it trades away the "properly...
with real login" part of the ask, in exchange for less setup.

**Ongoing maintenance/cost**: same as Option A's Supabase piece ($0, same
inactivity-pause caveat), minus the Google side entirely.

## Recommendation

Given the ask explicitly prefers "doing this properly with real login" over
faking it, **Option A** is the right fit for what was actually requested -
the added complexity over Option C is mostly front-loaded one-time Google
Cloud Console setup (~15-30 min), not ongoing burden, and it's the only
option that gives real per-person identity rather than a public,
guessable-ID convenience key. Option C is worth keeping in mind if the
Google Cloud Console setup turns out to be more friction than it's worth in
practice, or if a friend without a Google account ever comes up. Option B
doesn't meet the actual requirement and is included only because it was
asked for as an explicit comparison point.

Nothing here has been built. Waiting on which option to implement.
