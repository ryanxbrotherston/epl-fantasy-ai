# Next steps for Claude Code

This is a working repo, not a blank slate — read the code before planning
anything, especially `squad_optimizer.py` (the ILP formulation), `match_model.py`
(the Poisson goal model), and `fpl_api.py` (the live data contract everything
else depends on). The README's "Known issues" and "Not started yet" sections
are accurate as of this handoff.

## Immediate blocker

GW1 deadline is imminent. Until Ryan has created the real AI-run FPL account
and submitted a squad (using the AI Team tab's recommendation), nothing
downstream — the monitor, the "locked/live" AI Team mode, transfer planning —
has a real account to operate on. Don't build further on the assumption an
account exists; confirm it does first.

## What's built and backtested (as of this handoff)

- **Points model**: RandomForest, 6-season window (2019-20 to 2024-25),
  recency-weighted, tested against 2025-26 held out. MAE 0.991 vs 1.042 naive
  baseline. Extending to the full 9-season history was tried and tested (with
  and without recency weighting) - it didn't measurably help, see README.
- **Squad optimizer**: PuLP ILP, budget/position/club-limit constraints, picks
  full XI + captain + bench. Known limitation: optimizes average points, not
  ceiling - see README.
- **Match model**: Poisson goal model (attack/defence strength per team via
  `sklearn.linear_model.PoissonRegressor`, home advantage term, recency
  weighting). Backtested on 2025-26: 45.5% match-winner accuracy (vs 42.6%
  "always pick home team" baseline), Brier score 0.208 vs 0.218 naive. Modest
  but real edge - football is genuinely hard to predict, this isn't a
  breakthrough, it's a credible first pass. Derives match winner/draw/loss,
  exact-score grid, and clean-sheet probabilities from the same fitted lambdas.
  **Known simplification**: assumes home/away goals are independent Poisson
  processes - the real Dixon-Coles model adds a low-score correlation
  correction (0-0, 1-0, 0-1, 1-1) that this doesn't implement.
- **Player props**: anytime goalscorer/assist probabilities, built on top of
  the match model's fixture-difficulty lambda + each player's own rolling
  expected-goals/assists rate. Sanity-checked against real fixtures (Man City
  vs a weak side correctly skews heavily favorite; a fixture-adjusted premium
  striker's scoring probability comes out in a believable range) but not yet
  backtested with real calibration metrics the way the match model was -
  worth doing before trusting it for real decisions.

## Priority order for what's next

1. **Sanity-check the live layer for real.** Everything in `fpl_api.py` and
   `ai_team_monitor.py` was built and unit-tested against mocked/schema-accurate
   data (the sandbox that built this couldn't reach fantasy.premierleague.com).
   First real run should confirm: bootstrap-static shape matches what the code
   expects, entry/picks lookups work against a real Team ID, and the email
   actually sends via Gmail SMTP.
2. **The weekly predict → observe → recalibrate loop.** This is the core of
   what makes match_model.py and player_props.py actually useful in-season,
   not just a one-off backtest. Needs:
   - A script that runs before each gameweek's deadline: generates match/prop
     predictions for that week's fixtures, stores them (git-committed JSON/CSV
     is fine for now, same pattern as `data/alert_log.json` - a real DB isn't
     necessary yet at this data volume).
   - A script that runs after results are in: pulls actual results/scorers via
     the live FPL API, compares against what was stored, logs accuracy/Brier
     score for that week specifically (not just the one-time historical
     backtest - an ongoing, visible track record).
   - Periodic retraining of `match_model.py` incorporating this season's own
     now-completed matches, not just the pre-season historical window.
   - **This is the part that's genuinely blocked on something real**: player
     props need in-season rolling player form to actually update through the
     season, and that live rolling-feature engine doesn't exist yet (item 3
     below). Match-level retraining doesn't have this dependency and can be
     built now.
3. **Live in-season rolling player features.** Currently frozen at last
   season's closing form (`build_gw1_features.py`'s whole premise). The
   `element-summary/{player_id}/` endpoint's `history` field (this season's
   gameweek-by-gameweek data so far) is the fix - confirmed via documentation,
   not yet implemented. Unlocks: in-season transfer advice, live player props,
   and removes the "frozen at pre-season form" caveat on everything else.
4. **Fixture difficulty ticker.** Team strength data is already sitting in
   bootstrap-static; `match_model.py`'s fitted lambdas are arguably a *better*
   difficulty signal now than the raw strength ratings would be - worth using
   the match model's own output here instead of building something separate.
5. **Price change predictor.** Net transfers in/out per player is already in
   the data; standard approach is a momentum threshold, well documented in
   the FPL community, not novel.
6. **Risk/differential analyzer.** Ownership % is already in the data too —
   this is mostly a UI/framing exercise on data we already have.
7. **Multi-week transfer + chip planner.** The real lift. Needs a proper
   multi-period MILP (not a bigger version of the single-week ILP we have —
   genuinely different: decision variables per gameweek across a rolling
   5-6 week horizon, transfer-cost and banked-free-transfer accounting, chip
   usage-window constraints). Worth studying sertalpbilal's
   FPL-Optimization-Tools (open source, well-regarded in the FPL analytics
   community) as a reference formulation rather than deriving from scratch.
   Re-solve the full horizon on every run; only commit to the current
   gameweek's decision — everything further out should stay revisable as
   double/blank gameweeks get confirmed through the season.
8. **Deployment.** Push this repo to GitHub for real, connect it to Streamlit
   Community Cloud, set the `AI_TEAM_ID`/email secrets in both GitHub Actions
   and Streamlit Cloud's secrets manager (separate systems, both need it).
9. **Persistence for the Friends tab.** Currently session-only. Needed for
   any cross-visit leaderboard or history — Supabase is a reasonable default,
   not committed to.

## Known limitations, not yet solved

- The squad optimizer maximizes *average* predicted points, which structurally
  undervalues explosive/premium players (their ceiling matters more than their
  average, especially for captaincy) — see README for the concrete example.
  This needs a variance-aware objective, not a quick patch.
- The match model treats home/away goals as independent (no Dixon-Coles
  low-score correlation correction) and refits from a fixed pre-season
  historical window rather than adapting through the season yet (see priority
  2 above).
