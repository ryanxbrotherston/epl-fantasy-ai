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
- **Squad optimizer** (`squad_optimizer.py`): single-week ILP. Was
  undervaluing premiums; fixed with a price-conditional bias correction
  (`PRICE_BIAS_CORRECTION`) after backtesting showed the real cause was mean
  miscalibration, not missing variance-reward - see "Squad optimizer premium
  fix" below for the full story, including the variance approach that was
  tried first and backtested WORSE.
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
4. **Risk/differential analyzer.** DONE, but see caveat below - not
   spotless. Ready for review, not "ship blind."
5. **Deployment.** Push this repo to GitHub (should already be done by the
   time this is read), connect Streamlit Community Cloud, set the
   `AI_TEAM_ID`/email secrets in both GitHub Actions and Streamlit Cloud's
   secrets manager (separate systems, both need it).
6. **Persistence for the Friends tab.** Currently session-only. Supabase is
   a reasonable default, not committed to.

## Known limitations, not yet solved

- The match model treats home/away goals as independent (no Dixon-Coles
  low-score correlation correction) and refits from a fixed pre-season
  historical window rather than adapting through the season yet (see priority
  2 above).
- **Risk/differential analyzer's picks skew toward noisy low-minutes
  players** - predicted points for barely-used players are small-sample
  noise (same issue player_props.py had before its shrinkage fix), so
  ranking among them isn't very meaningful yet. Needs a minimum-minutes
  floor before a player enters the differential ranking at all.
- Backtesting risk_analyzer.py caught something worth remembering for any
  future feature that touches ownership: **low ownership is NOT free
  upside** - backtested against real 2025-26 data, low-ownership players
  score ~1 point/gameweek WORSE than high-ownership players at the same
  price. The crowd's ownership numbers are a genuinely informative signal,
  not noise to bet against by default.
