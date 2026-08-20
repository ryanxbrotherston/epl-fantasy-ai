# Next steps for Claude Code

This is a working repo, not a blank slate — read the code before planning
anything, especially `squad_optimizer.py`, `match_model.py`, and
`multi_week_planner.py`. The README's "Known issues" section is accurate.

## backtest_multi_week_planner.py (2026-08-20 session)

Priority 4. Closed the gap flagged in this file's own "multi_week_planner
re-verification" section above: no persisted, re-runnable backtest
existed for `multi_week_planner.py`/`multi_week_projections.py`, unlike
every other model in this repo - the original "tested against real
2022-23 DGWs" claim was ad-hoc and never saved. `src/backtest_multi_week_planner.py`
is that script now, run for real against GW18-26 of 2022-23 (the World
Cup fixture-congestion window) - **ALL CHECKS PASSED**, exit code 0:

1. A real double-gameweek club (Chelsea, GW19; Man City, GW20) projects
   HIGHER than its own single-fixture baseline that week - confirmed
   against real fixtures.csv data, not assumed.
2. A real blank-gameweek club (Brentford, GW25) projects EXACTLY ZERO
   that week.
3. The full multi-week ILP solves to Optimal across the whole horizon -
   no crash, no infeasibility.
4. Squad composition (15 players, position quotas, ≤3-per-club) stays
   valid every single week of the resulting plan.

**Deliberately not a points-model accuracy backtest**: 2022-23 is one of
`points_model.pkl`'s own training seasons (`train_points_model.py`'s
`TRAIN_SEASONS`), so treating it as held-out for point predictions would
be data leakage. Uses a model-independent points proxy instead (each
player's own trailing, backward-looking actual points-per-game that
season) - isolates what's actually being tested: does the *planner*
handle a real DGW/blank calendar correctly, not whether some point
estimate is accurate. `match_model.pkl` is still used for its fixture-
difficulty *scaling* role (relative multiplier, not a target prediction)
- same role it already plays, unflagged for leakage, in
`backtest_squad_optimizer.py`.

**Explicit, stated gap, not silently dropped**: chip timing isn't
re-validated here. No chip-eligibility-window data exists for a
historical season in the vaastav CSVs (that's live FPL account/season
metadata, not part of the historical dumps) - rather than guess plausible
windows, this runs with `chip_windows={}` and checks transfer/budget/
DGW/blank correctness only. The original ad-hoc test's "sensible chip
timing" was always just an impression, never itself checked against
ground truth - that's still true, just now written down instead of
implied.

## Overall rank display (2026-08-20 session)

Priority 3. Confirmed the exact field against a live response rather than
assuming: `entry/{team_id}/`'s `summary_overall_points` and
`summary_overall_rank`. Both are `null` right now for every manager
checked, including established ones with years of history - genuinely
FPL's real current state (nobody's overall rank/points exist until a
gameweek's been scored), not a bug.

Found and fixed a real latent bug while adding this: `render_team_lookup`
had `entry.get('summary_overall_rank', 'N/A'):,}` - `.get()`'s default
only applies when the key is *missing*, not when it's present but `None`,
and the key is always present (just `None` pre-season) - so this would
have raised `TypeError: unsupported format string passed to
NoneType.__format__` the first time it actually ran with real null-rank
data, which is happening right now. It was silently masked by an outer
`if entry.get("summary_overall_points"):` guard, but that's not a
guaranteed protection (points and rank aren't always in lockstep). Fixed
via a proper `format_rank()` helper that checks for `None` explicitly.

Added "Total points" and "Overall rank" metrics (via the new
`format_rank()` helper, `"N/A yet"` when null) alongside the existing
Bank/Team value in the AI Team locked/live mode and the shared
`render_team_lookup` (My Team + Friends). AI Team propose mode has no
real FPL account behind it, so nothing to add there. Verified against
live data (no crash, correct "N/A yet" display) and a full app boot.

## Predicted lineup integration (2026-08-20 session)

Priority 2. Ryan wanted a real predicted-lineup source (team news/press
conferences) instead of inferring starting likelihood purely from
historical minutes, with the early-prediction/official-confirmation
distinction preserved rather than collapsed into one signal. This took a
long research path with several real dead ends worth recording so nobody
re-treads them:

**Sources checked and why each was rejected, in order:**
- **Fantasy Football Scout** (the obvious, best-reputation source - real
  editors tracking press conferences): ToS (`/terms-and-conditions`,
  section 6.6) explicitly bans "mass, automated or systematic
  extractions... or use it to create or include it within another...
  electronic database" - exactly what an hourly automated pipeline is.
  Ryan explicitly chose to proceed anyway understanding the risk, but the
  actual page content turned out to be a second, independent blocker: the
  real per-club predicted XI is a **pitch-graphic image with empty alt
  text**, with only loose prose commentary around it - not machine-
  parseable into a clean flag without fragile NLP/OCR. Superseded before
  being built - see fpledits.com below.
- **RotoWire**: ToS restricts to "personal, non-commercial use", bans
  "reproduction... or creation of derivative works" - same category of
  problem as FFS, not used.
- **ESPN**: the specific article suggested was a one-off dated Week 1
  piece, not a durable per-gameweek source with a discoverable URL
  pattern - not viable for ongoing automation regardless of ToS.
- **Fantasy Football Pundit**: actively protected by Cloudflare
  bot-detection (confirmed: 403, `Server: cloudflare`, generic challenge
  page). This is a hard line, not a judgment call - bypassing bot
  detection is something this assistant won't build regardless of
  instruction, so this source was never in play once that was found.
- **API-Football**: real, documented API with an explicit
  `lineup_confirmed` field - exactly the predicted/confirmed distinction
  wanted. But its **free tier has no access to the current season at
  all** - confirmed live with a real key (`"Free plans do not have access
  to this season, try from 2022 to 2024"`), and confirmed this isn't a
  pre-season quirk (2025, a fully-completed past season, is *also*
  blocked - it's a permanent rolling ~2-year paywall, not a timing issue).
  Cheapest current-season tier: $19/month (Pro). Ryan chose not to pay for
  this once a free alternative worked out (see below).
- **LineupsLabs** and **fpledits.com's frontend pages**: both modern
  client-rendered Next.js apps where the actual lineup data loads via an
  in-browser fetch after page load, not present in the server-rendered
  HTML or the RSC payload (confirmed by inspecting both directly -
  LineupsLabs' own payload literally ships `"initialLeagueData":
  {"7":{"gameweeks":[]}}`, an empty placeholder). No browser access this
  session to inspect the real network request. Guessing conventional
  `/api/...` paths on LineupsLabs came back 404 across the board -
  stopped there rather than keep guessing, consistent with the "don't
  guess a URL" discipline this whole exercise was under. Ryan found
  fpledits.com's actual endpoint himself via his own browser's devtools
  Network tab - see below.

**What actually shipped**: `https://fpledits.com/api/predicted-lineup/1`
- a real, public, unauthenticated, structured JSON API returning all 20
clubs' predicted starting XIs in one request. ToS checked (`/terms`, RSC-
fetched since this site is also client-rendered for most pages but this
one's SSR'd): only prohibits "misuse, disruption, or unauthorized access"
- nothing restricting reading a public, intentionally-served endpoint.
`robots.txt` doesn't exist (404) - no crawl restriction either.

**Real data-quality bug found and worked around, not silently trusted**:
the endpoint's own `teamId`/`teamName` fields are stale, referencing an
old season's 20-club list (e.g. `teamId: 3` is labeled "Burnley", `19` is
"West Ham", `20` is "Wolves" - none of which are in the real current
2026-27 Premier League; the real current clubs in those slots are
Coventry City, Hull City, and Ipswich Town per FPL's own live
bootstrap-static). Verified this wasn't just a labeling issue by
cross-checking the actual player `id`s inside each mislabeled team's
`selectedLineup` against live bootstrap-static: every single player id
resolved correctly to their real current club (e.g. the "Burnley" entry's
11 players are, individually, Bournemouth's actual current squad).
`lineup_predictor.py` therefore **ignores `teamId`/`teamName` entirely**
and joins every player by FPL element id instead - sidesteps the bug
completely rather than trusting a broken label.

**A related endpoint, `/api/predicted-lineup/confirmed/1`, exists but
returns 401** (auth/paid feature) - confirms fpledits.com does track the
early/confirmed distinction internally, but the confirmed side isn't
usable without an account. Given the cost/complexity already spent
finding a working free source, Ryan chose to skip official
~60-75-min-pre-kickoff team-sheet confirmation automation entirely for
now, rather than pay for either this or API-Football's Pro tier. **This
is a real, acknowledged gap**: `lineup_predictor.py`'s "confirmed" basis
label currently only ever means "FPL's own official status field"
(injured/suspended/unavailable - a real, official designation, just not
the specific pre-kickoff-team-sheet kind originally asked for), never a
literal official starting-XI confirmation. Worth revisiting if a free
source turns up, or if the $19/mo or fpledits.com paid tier starts
feeling worth it once there's a track record to justify it.

**What's built**:
- `src/lineup_predictor.py`: `fetch_predicted_starting_ids()` (the early
  signal, with a documented None-on-failure contract so a source outage
  is never silently treated as "everyone's benched") and
  `starting_likelihood_flag()` (green/yellow/red per player, always
  labeled with which basis - `confirmed`/`early`/`none` - produced it).
- `app.py`: a "Starting?" column (🟢/🟡/🔴 + basis label) added to all
  three tabs' squad tables (AI Team propose mode, AI Team locked/live
  mode, My Team/Friends via the shared `render_team_lookup`), backed by a
  15-minute Streamlit cache (shorter than the hour-long cache on
  bootstrap/predictions, since this specific signal is meant to update
  through the week).
- `src/ai_team_monitor.py`: now checks the starting XI against the early
  signal too, not just official status. Bench-likely players (in the
  starting XI, no official status issue, but not in fpledits.com's
  predicted XI) get their own alert-log dedup key and their own clearly
  labeled email section ("EARLY WARNING - predicted only... could easily
  change before kickoff"), kept visibly separate from the "CONFIRMED -
  FPL's own official status" section for the pre-existing injury/
  suspension checks. Verified end-to-end with a synthetic squad including
  a real player (Saka) the source currently flags as bench-likely - both
  sections rendered correctly, dedup keys correct
  (`{element}_early_bench_likely` vs the existing `{element}_{status}`).
- `src/lineup_prediction_log.py` + a new `fpl_api.get_event_live()`:
  **backtesting this wasn't feasible** - fpledits.com only exposes its
  latest snapshot, no historical archive, so there's nothing to backtest
  against the way squad_optimizer.py/risk_analyzer.py's fixes were. Built
  the honest alternative instead, per instructions: `ai_team_monitor.py`
  now logs a snapshot of the predicted starting XI every hourly run
  (overwriting the same gameweek's entry, so the log ends up holding
  whatever was predicted closest to kickoff) and scores the *previous*
  gameweek's snapshot against real per-gameweek `minutes` (via the new
  `get_event_live()` - one request for the whole gameweek's actuals, not
  one per player) once results are in. This builds a real, growing
  accuracy track record over the season rather than a one-time claim -
  **there is no accuracy number to report yet**, since GW1 hasn't been
  played. `.github/workflows/ai_monitor.yml` updated to commit the two
  new log files back to the repo alongside the existing alert log (same
  reason: GitHub Actions runners don't persist anything otherwise).

**Not done, explicit gaps**: official pre-kickoff team-sheet confirmation
(see above - punted, not solved). Historical backtest of the early signal
(not feasible, logging forward instead, explicitly said so rather than
shipping an unvalidated number).

## BigBallsData evaluated and rejected for the confirmed-lineup gap (2026-08-20, later same night)

Ryan asked to try `api.bigballsdata.com` (published July 2026, single
maintainer, no track record - flagged as risky going in) to fill the
official-confirmation gap left above. Verified before building anything
on top of it, per instructions - **verification failed, nothing was
wired in**.

**What's real about it**: `/v1/matches` genuinely returns current-season
EPL data - real current clubs (cross-checked against live FPL data from
earlier tonight: Coventry City, Hull City, Ipswich Town, Sunderland all
correctly present), correct GW1 kickoff dates. Auth works
(`x-api-key`), ToS is clean (no restriction on automated polling - a
normal commercial API meant to be polled), ratelimit headers confirm a
genuine free-tier account.

**What isn't real about it - the actual thing needed**: the lineups
endpoint (`/v1/stored/matches/{id}/lineups`) returned `"available":
false` for **100/100 recently-finished football matches** checked across
every league they cover (mostly MLS, since football/EPL hasn't started
yet) - not a plan/paywall issue (clean 200s, lineups marketed as
free-tier), not a timing issue (these matches are *finished*, lineup data
should unambiguously exist by now). The response schema itself is a
second, independent red flag: `data.home`/`data.away` are typed as
generic `{field: string, value: any}` pairs in their own OpenAPI spec -
not a real player/position/status structure. Reads as a stubbed-out
feature the marketing copy describes ahead of what's actually been built,
not a working data source with a coverage gap.

**Decision**: confirmed-lineup automation stays manual for now (FPL's own
app shows this ~60-75 min pre-kickoff). The $19/mo API-Football Pro
option (see "Predicted lineup integration" above - real, working,
verified with a live key, just not free) is still sitting there if this
becomes worth paying for later. Nothing from BigBallsData was wired into
`ai_team_monitor.py` or anywhere else - no secrets added, no code
changed. **If BigBallsData's lineups coverage is ever reconsidered**,
re-check `/v1/stored/matches/{id}/lineups` against real *finished*
matches first (not just upcoming ones) before assuming their coverage has
caught up - that's the check that actually caught this, not the
matches-data check, which looked fine.

## Confirmed-lineup gap CLOSED: highlightly.net (2026-08-20, later still)

After BigBallsData (rejected above) and football-data.org (checked live
with a real key - legitimate service, but lineups require their paid
"Deep Data" tier, €29/mo minimum - free tier confirmed to return no
`lineups` key at all on a real finished match), Ryan found
`api.bigballsdata.com`'s TypeScript SDK snippet and a `fields=lineups`
query-param idea, neither of which changed anything (the SDK snippet only
exercised `/v1/matches`, never touched `lineups`; `fields=lineups` isn't
a real parameter - confirmed absent from BigBallsData's own OpenAPI spec,
and silently ignored when tried live - also surfaced a second, worse data
quality issue: a real Atlético Madrid vs Málaga fixture appeared twice
with two different final scores, not just duplicate metadata like the
earlier Hull City/Man Utd case). Then Ryan tried `highlightly.net`
(`soccer.highlightly.net`) - **this one is real and is now wired in.**

**Verified before building anything, same discipline as the rejected
candidates**: `/matches` returns real current-season clubs (Hull City,
Man City, Newcastle, Brentford, Ipswich, Brighton - all correct, no
staleness). ToS is clean for this use case (personal app, not a
competing database/reseller/gambling operation - all explicitly the only
things it restricts). Critically, tested the actual thing needed against
a real *finished* match (Leeds vs Everton, 2025-26 GW1) - not just an
upcoming one, since that's the check that caught BigBallsData's stub -
and got back genuine, richly-structured data: real players (Lucas Perri,
Pascal Struijk, Joe Rodon, ...), positions, formation, a separate
substitutes bench. Free tier includes lineups with no paywall (100
req/day) - the first candidate tonight where lineups aren't gated behind
a payment at all.

**A genuinely useful property, not just "it works"**: there's no separate
predicted mode. Tested against an unplayed GW1 fixture and got back an
empty/`"Unknown"`-formation response - so presence of real data *is* the
confirmation signal (per their docs, released ~30-40 min before kickoff,
once clubs confirm it). No boolean field to misinterpret, unlike
API-Football's `lineup_confirmed` flag design.

**What shipped** (`src/confirmed_lineup.py`, wired into
`ai_team_monitor.py` as a new third email section, "CONFIRMED - official
team sheet (highlightly.net)", sitting between the existing official-FPL-
status and early-fpledits-prediction sections in authority order):
- Two identity mismatches handled rather than trusted blind: **team
  names** differ from FPL's for 4 clubs (Man City/Manchester City, Man
  Utd/Manchester United, Spurs/Tottenham, Nott'm Forest/Nottingham
  Forest) - the mapping was built and verified by diffing both sources'
  real live 20-club lists directly, not guessed; the other 16 match via
  exact-string or substring comparison. **Player ids** are entirely
  different schemes between the two services - matched by name instead
  (first+second name, falling back to a surname/web_name check), with a
  3-state result (confirmed starting / confirmed benched / no confident
  match) specifically so a name-matching miss can never be mistaken for
  "not started."
- Rate-limit conscious by design: Highlightly's free tier is 100 req/day,
  and lineups aren't released until ~30-40 min before kickoff anyway, so
  polling every fixture every hour all week would waste the entire daily
  budget on empty responses. Only checks a given fixture once it's within
  2 hours of its own kickoff (`CONFIRMED_LOOKUP_WINDOW_HOURS`) - outside
  that window, no Highlightly call happens at all, same as if the source
  were unreachable.
- Graceful fallback throughout: an unreachable/not-yet-released source,
  an unmatched team, or an unmatched player all resolve to "no data for
  this check right now," never to a false "not started" - the existing
  two-tier system (official FPL status, then fpledits early prediction)
  is completely unaffected if Highlightly has nothing to say.
- Verified end-to-end with a synthetic squad exercising all three tiers
  together in one run (an officially-injured player, a confirmed-benched
  player via a faked imminent kickoff, and an early-warning-only player) -
  each landed in exactly the right email section, with the early-warning
  check correctly *not* re-flagging the confirmed-benched player.

**Real gap in the codebase now, not the feature**: the API key needs to
live in **GitHub Actions repository secrets**, not Streamlit Cloud's
Secrets manager - `ai_team_monitor.py` runs via GitHub Actions on its own
hourly cron, a completely separate secrets store from the Streamlit app
(the same distinction that already applies to `AI_TEAM_ID`/
`ALERT_EMAIL_*`). Ryan still needs to add `HIGHLIGHTLY_API_KEY` there
before this actually fires in production - the code is ready and
verified against live data, but unset until that secret exists.

**Not yet re-confirmed**: the real end-to-end path (an actual GW1 fixture
reaching its 2-hour pre-kickoff window, Highlightly having genuinely
released a lineup, and the email firing correctly against real - not
synthetic - squad data) hasn't happened yet, since GW1 hasn't kicked off
as of this session. Worth a real check once it has.

## Deployment status — CORRECTION (2026-08-20, later same day)

The "Immediate blocker" and "Deployment prep" sections originally written
earlier tonight (below, struck through in spirit — kept for the record,
not deleted) asserted the AI account didn't exist and that `AI_TEAM_ID`/the
Gmail App Password secret weren't set. **That was wrong, and Ryan corrected
it directly.** Deployment is live and confirmed working: the Streamlit
Cloud app is deployed (it had gone to sleep from inactivity, which is
normal Streamlit Community Cloud behavior on the free tier, and has since
been woken), and `AI_TEAM_ID` + the Gmail secret are both already set in
GitHub Actions secrets.

**The actual lesson, for this file and for whoever/whatever reads it
next**: this session had no way to check GitHub's Secrets UI, Streamlit
Cloud's dashboard, or whether a real FPL account existed under Ryan's
name — those all require either browser access this session didn't have,
or credentials it doesn't hold. The correct statement was always "I can't
verify this from here," not "this doesn't exist." Conflating the two
produced a confidently wrong claim in a document meant to be trusted at
face value later. Below, the original sections are kept but should be read
with that correction in mind — the concrete actions taken (GitHub push,
workflow file review) are still accurate; the claims about what didn't
exist yet are not.

## Immediate blocker (ORIGINAL — see correction above, status unverified from this session)

GW1 deadline is imminent (2026-08-21T17:30:00Z UTC — confirmed live from
bootstrap-static tonight). This section assumed the real AI-run FPL account
didn't exist yet; per the correction above, that was an unverified guess,
not a confirmed fact — the account may well already exist. Don't assume
either way from inside a session that can't check; ask Ryan or look for
direct evidence (e.g. `AI_TEAM_ID` being set, which it is) before building
on an assumption about account state.

## Deployment prep (2026-08-20 session) — ORIGINAL (see correction above)

Priority 3 tonight, as far as it could go without Ryan present:

- **Pushed to GitHub**: `origin/main` is now current (was 2 commits behind).
  Confirmed no secrets in the diff (`.streamlit/secrets.toml` is already
  gitignored) before pushing. This part stands, unaffected by the
  correction above.
- **`.github/workflows/ai_monitor.yml` reviewed**: correctly wired to read
  `AI_TEAM_ID`/`ALERT_EMAIL_*` from GitHub Actions secrets, on an hourly
  cron plus manual `workflow_dispatch`. Nothing to fix here. This also
  stands.
- ~~Streamlit Community Cloud connection, and setting secrets in both
  places, could NOT be done tonight — genuinely blocked~~ **Incorrect as
  written.** What was actually true: this session had no browser access to
  check share.streamlit.io or GitHub's Secrets UI, so it could not verify
  either was done. It was not true that they hadn't been done — both had
  been, before this session started. Confirmed directly by Ryan: Streamlit
  Cloud is deployed and live (was just asleep from inactivity), and
  `AI_TEAM_ID`/the Gmail secret are already in GitHub Actions secrets.
  Nothing left to do here as far as anyone in this thread knows.

## What's built and backtested (as of this handoff)

Everything below has real, honest metrics behind it — not vibes. See README
and each script's docstring for the numbers and what they mean.

- **Points model** (`train_points_model.py`): RandomForest, 6-season window,
  recency-weighted. MAE 1.012 vs 1.057 naive baseline (updated 2026-08-20
  after the `merged_gw.csv` duplicate-row fix - see below).
- **Squad optimizer** (`squad_optimizer.py`): single-week ILP. Was
  undervaluing premiums; fixed with a price-conditional bias correction
  (`PRICE_BIAS_CORRECTION`) after backtesting showed the real cause was mean
  miscalibration, not missing variance-reward - see "Squad optimizer premium
  fix" below for the full story, including the variance approach that was
  tried first and backtested WORSE. Refit 2026-08-20 (see "merged_gw.csv
  duplicate-row fix" below): +1.59 pts/GW now, down from +2.24 - real effect,
  weaker than first measured.
- **Match model** (`match_model.py`): Poisson goal model. 45.5% winner
  accuracy vs 42.6% naive, Brier 0.208 vs 0.218. Modest real edge. Now has an
  optional Dixon-Coles low-score correlation correction (`fixture_probabilities(...,
  rho=...)`, rho fit by MLE in `fit_dixon_coles_rho`) - see "Dixon-Coles" below
  for why the backtested effect is real but tiny for this data.
- **Player props** (`player_props.py`): anytime scorer/assist probabilities.
  Was initially WORSE than naive (overconfident from a flat 75-min assumption
  and no shrinkage on noisy small-sample rates) - fixed, now beats baseline
  on both scorer (0.0276 vs 0.0302 Brier) and assist (0.0284 vs 0.0294 Brier)
  (updated 2026-08-20 after the `merged_gw.csv` duplicate-row fix).
- **Fixture ticker** (`fixture_ticker.py`): built directly on match_model's
  lambdas, not a separate heuristic. Sanity-checked, not separately
  backtested (low marginal need given it's a direct derivative).
- **Price predictor** (`price_predictor.py`): weekly resolution only - see
  its docstring for why (FPL's real algorithm is daily, undocumented; our
  historical data is weekly snapshots). AUC 0.87 (rise) / 0.76 (fall),
  6.7x / 4.2x lift in the top-decile watch list vs base rate.
- **Live rolling features** (`live_rolling_features.py`): pulls this
  season's actual gameweek-by-gameweek player history via
  `element-summary/{id}/`, replacing the frozen pre-season proxy. Validated
  against schema-accurate mock data only (live API unreachable from the
  sandbox that built this) - caught and fixed a real bug this way: FPL's
  live API returns some stats (ict_index, threat, creativity, influence) as
  quoted strings, not numbers. Confirm against a real gameweek once GW1+
  exists.
- **Multi-week transfer + chip planner** (`multi_week_planner.py` +
  `multi_week_projections.py`): real multi-period MILP, not a bigger
  single-week ILP - per-gameweek squad evolution via explicit transfer
  variables, free-transfer banking (capped at 5), chip eligibility windows
  pulled from bootstrap-static (not hardcoded), hit costs. Tested against
  REAL double/blank gameweeks in the 2022-23 season (World Cup fixture
  congestion) - correctly spikes projected points and transfer activity
  around confirmed DGWs, sensible wildcard/free-hit/triple-captain timing.
  Two real bugs found and fixed during testing: a nonlinear objective term
  (PuLP can't multiply two decision variables - needed proper AND-
  linearization for the triple-captain interaction) and self-cancelling
  buy/sell pairs the solver would pick when the transfer budget had slack
  (net-zero effect, but a confusing plan to hand someone).
  **Known simplification**: the per-week budget constraint doesn't yet
  distinguish buy price from sell price (FPL takes a cut on profit when you
  sell a risen player) - fine for a first pass, worth tightening.
  **2026-08-20**: no persisted backtest script exists for the 2022-23 DGW
  claim above (it was ad-hoc, never saved) - see "multi_week_planner
  re-verification" below. Also found and fixed a real bug on the first live
  end-to-end run of the full pipeline: `build_projection_table()` didn't
  carry `now_cost` through, which `plan()` needs for its budget constraint.

## Squad optimizer premium fix (2026-08-19 overnight session)

Priority 1 tonight was the "optimizes average, undervalues explosive
premiums" limitation flagged repeatedly above. Worth documenting the full
path here since the fix that shipped is NOT the one the framing above
suggested, and re-trying the obvious idea would waste a future session.

**Tried first: variance/ceiling reward.** Built `player_points_variance()`
on `player_props.py`'s existing Poisson framework (goal_lambda/assist_lambda
from a player's own rolling rate + clean-sheet probability from their
rolling goals-conceded rate), then added `risk_weight * points_std` on top
of `predicted_points` as the ILP objective, plumbed through
`pick_squad`/`pick_starting_xi`. Backtested honestly against 2025-26 with a
fresh-pick-each-gameweek method (real `pick_squad`/`pick_starting_xi`,
actual XI+captain points scored that week): **it made things worse**, and
monotonically so - avg actual points/GW fell from 60.6 at risk_weight=0.0 to
56.9 at risk_weight=1.0, 0/32 gameweeks improved by any nonzero weight.
Why, in hindsight: picking a *fixed-size* squad (15 players, fixed position
quotas) under a *linear* points payoff is an expected-value-maximization
problem. A uniform-cardinality selection with a linear objective has no
convexity for variance to exploit - rewarding it just trades away real
expected points for nothing. Ceiling/variance only pays off when the payoff
itself is convex (rank in a league table, "I need a differential to catch
up") - which is exactly the axis `risk_analyzer.py` already models
separately, not something squad selection's own objective should chase.

**What was actually wrong:** checked `predicted_points` against real actual
points by price tier on the held-out season - the model+blend systematically
UNDER-predicts expensive players specifically:

| price tier | mean bias (actual − predicted), full 2025-26 |
|---|---|
| budget (<£5.0m) | +0.13 pts/GW |
| mid (£5.0-7.0m) | +0.56 pts/GW |
| premium (£7.0-9.0m) | +1.19 pts/GW |
| elite (£9.0m+) | +1.35-1.6 pts/GW |

Most likely cause: `RandomForestRegressor`'s usual behavior of pulling
extreme predictions toward the training distribution's bulk
(`min_samples_leaf=5`, `max_depth=10` both regularize hard), diluted further
by blending in `ep_next`/`ppg` that don't fully correct it. This is a
calibration problem, not a variance problem.

**What shipped:** `squad_optimizer.PRICE_BIAS_CORRECTION` - a linear fit of
that bias against price, added onto `predicted_points`. Important subtlety:
a *uniform* rescale of predicted_points (e.g. the naive "just multiply
everyone's score by 1.4") is a no-op for this ILP - since squad size and
position quotas are fixed cardinalities, any affine transform applied to
every player equally cannot change which combination is optimal. The
correction has to be *price-conditional* to actually change who gets picked,
which is what makes this a real fix rather than a cosmetic one.

**Backtest** (`backtest_squad_optimizer.py`): correction fit on GW6-20 of
2025-26, evaluated fresh-pick-each-week on GW21-37 (never seen by the fit) -
+2.24 actual XI+captain points/GW on average (62.9 vs 60.7 baseline),
improved in 10/17 held-out gameweeks.

**Not fully closed:** re-running the same risk_weight sweep on TOP of the
price-corrected baseline (still on GW21-37, so tuned and evaluated on the
same data - not a clean held-out test) showed a further small lift at
risk_weight≈0.15 (~63.8 vs 62.9) before degrading again at higher weights.
Real, but not validated on data separate from what picked it - didn't ship
it for that reason. Worth a proper three-way split (fit price-correction /
tune risk_weight / evaluate) if a future season's data makes that feasible
without starving each split of gameweeks.

**Also found, not fixed:** `merged_gw.csv` (2025-26) has (element, GW)
duplicate rows of two different kinds - exact duplicate rows (a data-file
glitch) and genuine double-gameweek rows (two different fixtures, should
sum not duplicate). `backtest_squad_optimizer.py` dedupes/aggregates before
building rolling features; `train_points_model.py` and
`backtest_player_props.py` do NOT, so the same distortion (a "trailing 5
gameweeks" window that's silently shorter, or double-counted, around any
affected player) is present in the shipped points model and its backtested
MAE. Likely small in aggregate (rare rows) but not verified - worth checking
before trusting those exact numbers to the decimal.

## Risk analyzer minutes floor (2026-08-19 overnight session)

Priority 2 tonight: `risk_analyzer.analyze()` now drops any player below
`MIN_ROLL_MINUTES` (60 - FPL's own threshold for the full 2-point appearance
bonus, not an arbitrary number) before ranking anyone, instead of ranking
the full pool including fringe players whose `predicted_points` is mostly
small-sample noise.

**Backtest** (`backtest_risk_analyzer_minutes_floor.py`): for each 2025-26
gameweek, ran `analyze(..., target_rank_direction="chasing")` with and
without the floor and compared the actual points earned by
`top_picks_by_position()`'s recommendations that week. No floor: 1.60 actual
pts/pick average. floor=60: 3.42 pts/pick, improved in 31/32 gameweeks - a
much cleaner result than the squad-optimizer fix above, since filtering
obvious small-sample noise is a much less subtle problem than a calibration
bias. Stricter floors (90 min) score even higher on this narrow metric, but
that starts filtering out genuine rotation-player differentials rather than
just noise, so 60 was kept as the principled cutoff rather than the metric-
maximizing one.

**Re-verified 2026-08-20** against the retrained (post-dedup-fix) points
model - same script, same held-out season/GW range, no code changes needed
(`backtest_risk_analyzer_minutes_floor.py` already builds its own rolling
features from `backtest_squad_optimizer.py`'s helpers, which already use the
new model). Result: **essentially unchanged, if anything slightly stronger**:

| | before | after (retrained model) |
|---|---|---|
| floor=0 (no filter) | 1.60 pts/pick | 1.438 pts/pick |
| floor=60 (shipped) | 3.42 pts/pick | 3.443 pts/pick |
| mean lift | +1.82 | +2.005 |
| GWs improved (of 32) | 31 | 31 |

Unlike the squad-optimizer price-bias correction (which shifted
meaningfully when the model was retrained - see below), this fix is robust
to it. Makes sense: the minutes floor is a hard filter on `roll_minutes`
(observed playing time, not a model prediction), not a correction tuned
against the model's specific bias pattern - it doesn't care what the exact
`predicted_points` values are, only whether a player has enough of a track
record for ranking among them to mean anything. Nothing to change here.

## Dixon-Coles correction (2026-08-19 overnight session)

Implemented per the standard Dixon-Coles (1997) approach: a correction
factor tau(x,y) applied to the four low-score cells (0-0, 1-0, 0-1, 1-1) of
the independent-Poisson score grid, controlled by a single correlation
parameter rho, fit by MLE holding lambda_home/lambda_away fixed from the
already-fitted Poisson regression (`fit_dixon_coles_rho` in `match_model.py`).
rho is now saved in `models/match_model.pkl` alongside the model/encoder.
`fixture_probabilities(lambda_home, lambda_away, rho=...)` defaults to
rho=0.0 (old behavior) so nothing downstream broke by adding the parameter.

**Backtest result** (`backtest_match_model.py`, now runs both variants
side by side): fitted rho on the 6 training seasons = **-0.01** - real, but
an order of magnitude smaller than the original Dixon-Coles paper's typical
range (-0.1 to -0.2). On the held-out 2025-26 season this makes almost no
practical difference: match-winner Brier 0.2078 → 0.2077, draw Brier
0.1997 → 0.1996, clean-sheet and goals-MAE identical to 3 decimal places.
Implemented correctly and available (`rho` is in the saved model bundle for
any future caller that wants it), but the "documented simplification" this
was meant to fix turns out not to be costing much in practice for this
dataset/model - modern Premier League scoring, pooled across 6 recency-
weighted seasons, just doesn't show the low-score correlation the original
1990s data did. Nothing currently calls `fixture_probabilities()` outside
the backtest (it's not wired into the live app yet), so this was a
contained, low-risk change.

## multi_week_planner re-verification (2026-08-20 session)

Checked whether last night's "tested against real 2022-23 double/blank
gameweeks" claim for `multi_week_planner.py`/`multi_week_projections.py`
still holds after the `merged_gw.csv` dedup fix retrained `points_model.pkl`.
**Could not literally "re-run the existing backtest" - there isn't one.**
Unlike `squad_optimizer.py`/`risk_analyzer.py`, there's no
`backtest_multi_week_planner.py` in the repo (checked: grepped the whole
tree for `multi_week_planner`/`multi_week_projections`, only the module
files and this doc reference them). The 2022-23 DGW validation was done
ad-hoc last session and never saved as a script, so there's no numeric
headline metric to compare before/after either - the original claim was
qualitative ("correctly spikes... sensible timing"), not a pts/GW number
like the other two fixes. **Worth writing an actual
`backtest_multi_week_planner.py` at some point so this doesn't keep
happening** - didn't do it tonight since it wasn't the ask, but flagging it
as a real gap, same category as the missing dedup fix was.

What was actually checked instead, both against real data:

1. **Is the DGW/blank-fixture logic even coupled to `points_model.pkl`?**
   No - by construction. `multi_week_projections.build_projection_table()`
   takes `predicted_points` as an opaque per-player number and either sums
   it across however many fixtures a team has that gameweek (adjusted by
   `match_model`'s fixture-difficulty multiplier) or zeroes it if there are
   none - it never looks at how that number was produced. A retrained
   points model literally cannot change this behavior. Verified this isn't
   just a code-reading claim: pulled 2022-23's real fixtures.csv, confirmed
   real DGWs independently from the data (Chelsea and Fulham both play
   twice in real GW19, Man City/Man Utd/Spurs/Crystal Palace in real GW20)
   and a real blank (Brentford, GW25), ran them through the current
   `build_projection_table()` with a flat dummy `predicted_points=5.0` for
   every player - Chelsea GW19 correctly spiked to 9.16, Man City GW20 to
   14.99, Brentford GW25 came out exactly 0.0. Structurally intact,
   independent of any model retrain, past or future.

2. **Does the full pipeline still run end-to-end with the retrained model?**
   This surfaced a real, separate bug, unrelated to the retrain:
   `multi_week_planner.plan()` requires a `now_cost` column on
   `projection_table` for its budget constraint, but
   `build_projection_table()` never carried `now_cost` through from
   `squad_predictions` in the first place - a `KeyError` on the very first
   live smoke test of the full pipeline (`load_predictions()` →
   `build_projection_table()` → `plan()`), meaning this exact path may
   never have actually been run end-to-end as committed. Fixed by carrying
   `now_cost` through unmodified (not fixture-adjusted, unlike
   `predicted_points`) in `build_projection_table()`, matching the existing
   pattern for `id`/`web_name`/`position`/`team_name`. After the fix, ran
   the full pipeline for real against live current-season data (528
   available players, GW1-5, real chip windows from bootstrap-static) with
   the retrained model: solved successfully, produced a sane 5-week plan
   (consistent Haaland captaincy, wildcard in GW2, free hit in GW4, triple
   captain in GW5 - plausible chip spread for a fresh-account opening
   horizon, though this is a live-data smoke test, not a re-run of the
   qualitative 2022-23 chip-timing check, which would need that season's
   real chip-window data to repeat properly).

**Bottom line**: the part of the original claim that's mechanically
guaranteed (DGW sum / blank zero) is confirmed still correct and provably
un-affected by any model retrain. The part that does depend on
`predicted_points` (which players/chips the ILP actually picks) has no
persisted before/after to compare, same situation as the squad-optimizer
correction until it's backtested with real numbers - the pipeline runs and
solves correctly now (it didn't, until the `now_cost` fix), but "sensible
chip timing" hasn't been re-validated against real fixture congestion the
way it was last night.

## Live API sanity check (2026-08-20 session)

Priority 1 tonight. This machine (unlike the sandbox that built the live
layer) can actually reach `fantasy.premierleague.com` — ran every live call
for real, no mocking:

- `bootstrap-static/`: 200 OK, 595 elements, 38 events. `current_gameweek()`
  correctly falls to `is_next` (GW1, deadline 2026-08-21T17:30:00Z — the
  season hasn't started yet, confirmed via `current`/`is_current` both False
  as expected pre-deadline).
- `entry/{id}/`, `entry/{id}/history/`, `fixtures/?event=1`,
  `element-summary/{id}/`: all confirmed against real IDs, schema matches
  what the code assumes. Invalid team IDs return `None` cleanly (no crash).
- `entry/{id}/event/{gw}/picks/`: correctly returns `None` (404) for every
  team right now, because GW1 hasn't happened yet — there are no picks to
  read anywhere in the live API yet, for anyone. This is real pre-season
  state, not a bug. **Re-confirm this specific endpoint once GW1 has
  actually been played** — it's the one call that couldn't be exercised
  with real data tonight.
- `ict_index`/`threat`/`creativity`/`influence` are indeed quoted strings on
  live data, confirming the bug the previous session caught defensively —
  `pd.to_numeric(..., errors="coerce")` in `live_rolling_features.py` and
  `ai_team_monitor.py` handles it correctly, zero coercion failures.
- `ai_team_monitor.py` ran end-to-end (`AI_TEAM_ID=1 python
  src/ai_team_monitor.py`) against live data and printed the correct
  pre-deadline message without error. Email sending itself wasn't exercised
  (no real Gmail App Password in this session) — untested still.
- `app.py` boots under real `streamlit run` and serves HTTP 200 against live
  data. Couldn't visually drive the tabs — no browser automation available
  in this session — but every underlying live-data call the UI makes
  (`current_gameweek`, `team_picks_dataframe`, `get_entry`) was exercised
  directly above.

**Real bug found and fixed**: `ai_team_monitor.flag_problem_players()`
declared `DOUBTFUL_THRESHOLD = 50` and its docstring claimed doubtful
players below that threshold get flagged, but the code never actually
checked `chance_of_playing_next_round` — and `fpl_api.team_picks_dataframe()`
didn't even carry that column through its merge, so there was nothing to
check. Confirmed live: 28 players currently have status `'d'` (doubtful),
24 at 75% (correctly fine) and 2 at 25% (should have been flagged, weren't).
Fixed both — merge now includes `chance_of_playing_next_round`, and
`flag_problem_players` actually applies the threshold. Verified against
live data: the 2 sub-50% players are now flagged, the 24 at 75% still
aren't. No live squad exists yet to regression-test this against a real
alert email (GW1 hasn't happened), so re-confirm once the AI account has a
real squad and a real doubtful player shows up.

## merged_gw.csv duplicate-row fix (2026-08-20 session)

Priority 2 tonight. Applied the same dedup/aggregate fix
`backtest_squad_optimizer.py` already had (drop exact-duplicate rows, then
`groupby(["element", "GW"]).agg(sum for CORE_STATS+OPTIONAL_STATS, first for
everything else)` before building any rolling window) to
`train_points_model.py`'s `load_season()` and `backtest_player_props.py`'s
`build_test_season_rolling()`. Re-ran both end-to-end (real run, not
estimated) and compared:

| | before (had the bug) | after (fixed) |
|---|---|---|
| Points model MAE, held-out 2025-26 | 0.993 | **1.012** |
| Naive rolling-avg baseline MAE | 1.042 | 1.057 |
| Scorer Brier | 0.0274 (vs naive 0.0300) | 0.0276 (vs naive 0.0302) |
| Assist Brier | 0.0280 (vs naive 0.0290) | 0.0284 (vs naive 0.0294) |

Headline claims still hold (model still clearly beats naive baseline on all
three metrics, by a similar margin) but the exact numbers in the README are
now stale in the fourth-decimal sense the previous session flagged as
unverified - **README's "MAE 0.99" / player-props Brier numbers should be
updated to the "after" column above.** Train/test row counts dropped
~4-5% (156207→149362 train, 29757→29338 test rows pre-rolling) confirming
the duplication was real and not negligible, just not large enough to flip
any headline conclusion.

**Cascading effect on the squad optimizer (found, not just anticipated):**
retraining `points_model.pkl` changes its predictions, and
`squad_optimizer.PRICE_BIAS_CORRECTION` is a hardcoded linear fit *against
this specific model's* bias pattern - so it went stale the moment the model
was retrained. Refit it the same way the original was derived (full 2025-26
season, `bias = actual - predicted` vs price, `np.polyfit`) using the new
model: `{"a": -0.578167, "b": 0.018724}`, replacing the old
`{"a": -0.776184, "b": 0.020478}`. Re-ran `backtest_squad_optimizer.py`
(which fits its own GW6-20 correction independently for its held-out
methodology, unaffected by the production constant) with the new model:

| | before | after |
|---|---|---|
| Uncorrected baseline, avg XI+captain pts/GW | 60.7 | 57.47 |
| Price-corrected, avg XI+captain pts/GW | 62.9 | 59.06 |
| Mean lift | +2.24 | **+1.588** |
| GWs improved (of 17) | 10 | **7** |

The fix is still net positive (total points across the 17-GW eval window:
1004 vs 977 uncorrected) but the story is meaningfully weaker than last
night's - lift shrank by ~30% and it now improves fewer than half the
held-out gameweeks rather than well over half. Worth a closer look before
leaning on this number for a real decision (e.g. does the correction need a
different functional form now, or is the weaker result mostly just fewer
duplicate-inflated premium-player rows propping up the old fit).

**Update 2026-08-20**: both re-verified now. `risk_analyzer.py`'s
minutes-floor backtest holds essentially unchanged against the retrained
model (see "Risk analyzer minutes floor" above for the numbers) - it's a
hard filter on observed minutes, not a fit against the model's bias
pattern, so it was never really exposed the way the price correction was.
`multi_week_planner.py` has no persisted backtest to re-run in the first
place (see "multi_week_planner re-verification" below) - confirmed instead
that its DGW/blank handling is structurally independent of the points
model (provably can't regress from a retrain) and found+fixed an unrelated
`now_cost` plumbing bug on the first real end-to-end run of the full
pipeline.

`models/points_model.pkl` and `models/feature_columns.json` are
regenerated and committed with this fix.

## Priority order for what's next

1. **Sanity-check the live layer for real.** DONE above, except the picks
   endpoint (blocked on GW1 actually happening) and the real email send
   (blocked on Gmail App Password secrets being set).
2. **Wire the multi-week planner into live data.** Currently tested against
   historical seasons only (2022-23's real double/blank gameweeks). Needs:
   `fpl_api`'s live entry (current squad, bank, free transfers),
   bootstrap-static's real chip list and which chips are already used, and
   `multi_week_projections.py` fed from `live_rolling_features.py` instead
   of the frozen pre-season proxy once step 1 is confirmed working.
3. **The weekly predict → observe → recalibrate loop.** A script that runs
   before each deadline (generate + store predictions from match_model.py,
   player_props.py, and the planner), and one that runs after (pull actual
   results via the live API, log accuracy for that week specifically - an
   ongoing visible track record, not just the one-time historical backtest).
   Periodic retraining of match_model.py incorporating this season's own
   now-finished matches. Git-committed JSON/CSV logging is fine for now,
   same pattern as `data/alert_log.json`.
4. **Risk/differential analyzer.** DONE, including the minimum-minutes floor
   (see "Risk analyzer minutes floor" below). Ready for review, not
   "ship blind."
5. **Deployment.** Push this repo to GitHub (should already be done by the
   time this is read), connect Streamlit Community Cloud, set the
   `AI_TEAM_ID`/email secrets in both GitHub Actions and Streamlit Cloud's
   secrets manager (separate systems, both need it).
6. **Persistence for the Friends tab.** Currently session-only. Supabase is
   a reasonable default, not committed to.

## Known limitations, not yet solved

- The match model refits from a fixed pre-season historical window rather
  than adapting through the season yet (see priority 2 above).
- Backtesting risk_analyzer.py caught something worth remembering for any
  future feature that touches ownership: **low ownership is NOT free
  upside** - backtested against real 2025-26 data, low-ownership players
  score ~1 point/gameweek WORSE than high-ownership players at the same
  price. The crowd's ownership numbers are a genuinely informative signal,
  not noise to bet against by default.
