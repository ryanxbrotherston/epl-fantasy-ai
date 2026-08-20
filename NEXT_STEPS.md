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

**Not re-verified tonight, flagged for whoever picks this up next**:
`risk_analyzer.py` and `multi_week_planner.py`/`multi_week_projections.py`
both consume `squad_optimizer.load_predictions()`, so they're downstream of
tonight's retrained model and refit correction too. `risk_analyzer.py`'s
minutes-floor backtest (`backtest_risk_analyzer_minutes_floor.py`) and
`multi_week_planner.py`'s DGW test haven't been re-run against the new
model - their previously-reported numbers may also be slightly stale, same
mechanism as the squad optimizer above. Didn't chase this further tonight
to stay inside the stated priority order; flagging rather than guessing.

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
