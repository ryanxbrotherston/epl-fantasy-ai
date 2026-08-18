# Next steps for Claude Code

This is a working repo, not a blank slate — read the code before planning
anything, especially `squad_optimizer.py` (the ILP formulation) and
`fpl_api.py` (the live data contract everything else depends on). The
README's "Known issues" and "Not started yet" sections are accurate as of
this handoff.

## Immediate blocker

GW1 deadline is imminent. Until Ryan has created the real AI-run FPL account
and submitted a squad (using the AI Team tab's recommendation), nothing
downstream — the monitor, the "locked/live" AI Team mode, transfer planning —
has a real account to operate on. Don't build further on the assumption an
account exists; confirm it does first.

## Priority order for what's next

1. **Sanity-check the live layer for real.** Everything in `fpl_api.py` and
   `ai_team_monitor.py` was built and unit-tested against mocked/schema-accurate
   data (the sandbox that built this couldn't reach fantasy.premierleague.com).
   First real run should confirm: bootstrap-static shape matches what the code
   expects, entry/picks lookups work against a real Team ID, and the email
   actually sends via Gmail SMTP.
2. **Fixture difficulty ticker.** Cheapest win available — team strength data
   (attack/defence, home/away) is already sitting unused in bootstrap-static.
3. **Price change predictor.** Net transfers in/out per player is already in
   the data; standard approach is a momentum threshold, well documented in
   the FPL community, not novel.
4. **Risk/differential analyzer.** Ownership % is already in the data too —
   this is mostly a UI/framing exercise on data we already have.
5. **Multi-week transfer + chip planner.** The real lift. Needs a proper
   multi-period MILP (not a bigger version of the single-week ILP we have —
   genuinely different: decision variables per gameweek across a rolling
   5-6 week horizon, transfer-cost and banked-free-transfer accounting, chip
   usage-window constraints). Worth studying sertalpbilal's
   FPL-Optimization-Tools (open source, well-regarded in the FPL analytics
   community) as a reference formulation rather than deriving from scratch.
   Re-solve the full horizon on every run; only commit to the current
   gameweek's decision — everything further out should stay revisable as
   double/blank gameweeks get confirmed through the season.
6. **Deployment.** Push this repo to GitHub for real, connect it to Streamlit
   Community Cloud, set the `AI_TEAM_ID`/email secrets in both GitHub Actions
   and Streamlit Cloud's secrets manager (separate systems, both need it).
7. **Persistence for the Friends tab.** Currently session-only. Needed for
   any cross-visit leaderboard or history — Supabase is a reasonable default,
   not committed to.

## Known limitation, not yet solved

The squad optimizer maximizes *average* predicted points, which structurally
undervalues explosive/premium players (their ceiling matters more than their
average, especially for captaincy) — see README for the concrete example.
This needs a variance-aware objective, not a quick patch. Worth a real design
pass, not a bolt-on.
