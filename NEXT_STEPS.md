# Next steps for Claude Code

This is a working repo, not a blank slate — read the code before planning
anything, especially `squad_optimizer.py`, `match_model.py`, and
`multi_week_planner.py`. The README's "Known issues" section is accurate.

## Immediate blocker

GW1 deadline is imminent. Until Ryan has created the real AI-run FPL account
and submitted a squad (using the AI Team tab's recommendation), nothing
downstream — the monitor, the "locked/live" AI Team mode, transfer planning —
has a real account to operate on. Don't build further on the assumption an
account exists; confirm it does first.

## What's built and backtested (as of this handoff)

Everything below has real, honest metrics behind it — not vibes. See README
and each script's docstring for the numbers and what they mean.

- **Points model** (`train_points_model.py`): RandomForest, 6-season window,
  recency-weighted. MAE 0.991 vs 1.042 naive baseline.
- **Squad optimizer** (`squad_optimizer.py`): single-week ILP. Known
  limitation: optimizes average points, not ceiling (undervalues premiums).
- **Match model** (`match_model.py`): Poisson goal model. 45.5% winner
  accuracy vs 42.6% naive, Brier 0.208 vs 0.218. Modest real edge.
- **Player props** (`player_props.py`): anytime scorer/assist probabilities.
  Was initially WORSE than naive (overconfident from a flat 75-min assumption
  and no shrinkage on noisy small-sample rates) - fixed, now beats baseline
  on both scorer (0.0274 vs 0.0300 Brier) and assist (0.0280 vs 0.0290).
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

## Priority order for what's next

1. **Sanity-check the live layer for real.** `fpl_api.py`, `ai_team_monitor.py`,
   and `live_rolling_features.py` are all built and unit-tested against
   mocked/schema-accurate data only. First real run should confirm:
   bootstrap-static shape, entry/picks lookups, element-summary history,
   and the email alert actually sending.
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
4. **Risk/differential analyzer.** Ownership % is already in the data -
   mostly a UI/framing exercise on data already available, lowest-effort
   item left on the list.
5. **Deployment.** Push this repo to GitHub (should already be done by the
   time this is read), connect Streamlit Community Cloud, set the
   `AI_TEAM_ID`/email secrets in both GitHub Actions and Streamlit Cloud's
   secrets manager (separate systems, both need it).
6. **Persistence for the Friends tab.** Currently session-only. Supabase is
   a reasonable default, not committed to.

## Known limitations, not yet solved

- The squad optimizer maximizes *average* predicted points, which structurally
  undervalues explosive/premium players (their ceiling matters more than their
  average, especially for captaincy) — see README for the concrete example.
  This needs a variance-aware objective, not a quick patch.
- The match model treats home/away goals as independent (no Dixon-Coles
  low-score correlation correction) and refits from a fixed pre-season
  historical window rather than adapting through the season yet (see priority
  2 above).
