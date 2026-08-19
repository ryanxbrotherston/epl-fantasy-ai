# EPL Fantasy AI

Same architecture as the NRL prediction dashboard: historical data → rolling-form
features → RandomForestRegressor → decision layer on top (there, a Multi Builder;
here, a squad/transfer optimizer). Built for the 2026/27 FPL season.

## What's working

- **`app.py`** — Streamlit app, three tabs:
  - **AI Team**: the model's own squad/XI/captain, fully automated
  - **My Team**: enter your FPL Team ID, pulls your real squad (public data, no login)
  - **Friends**: same lookup, unlimited friends, session-based for now
- **`src/fpl_api.py`** — live FPL API connector (bootstrap-static, entry lookup,
  gameweek picks, history, fixtures). No auth needed — this is the same public
  data FPL's own mini-leagues use to show everyone else's teams.
- Objective function fixed: now blends the trained model with FPL's own
  `ep_next` estimate and last season's points-per-game, and hard-excludes
  unavailable/injured players. Full £100.0m budget now gets used (was leaving
  ~£8m on the table before).

### Premium players: fixed, but not the way it looked at first

The optimizer maximizes predicted points, and premiums were getting
underweighted — but backtesting the obvious fix (reward variance/ceiling, see
NEXT_STEPS.md for the full negative result) showed that's not actually the
problem: picking a fixed-size squad under a linear points payoff is an
expected-value problem, not a portfolio one, so rewarding variance just
traded away real points for nothing. The real issue, found by checking
predicted points against actual points by price tier on held-out data: the
model+blend systematically **under-predicts expensive players specifically**
(bias ~+0.13 pts/GW at budget prices, growing to ~+1.5 pts/GW at elite
prices) — a calibration problem. `squad_optimizer.py`'s
`PRICE_BIAS_CORRECTION` fixes it with a price-conditional correction, fit on
one part of the 2025-26 season and validated on a held-out later part it
never saw: +2.24 actual XI+captain points/GW on average, improved 10/17
gameweeks. See `backtest_squad_optimizer.py`.

### Testing caveat

This sandbox can't reach `fantasy.premierleague.com` (not on the network
allowlist), so `fpl_api.py`'s live calls are validated against documented
schema from multiple independent sources, not tested end-to-end in this
environment. Everything else (model training, squad optimizer, Streamlit
rendering) has been run and verified locally. **First live run — on your
machine or once deployed — should be sanity-checked against your own real
Team ID.**

## Previously working

- **`src/train_points_model.py`** — trains a RandomForestRegressor on 3 seasons
  (2022-23 to 2024-25) of gameweek-by-gameweek player data from the
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
  dataset, using a trailing 5-gameweek rolling window per player (minutes, points,
  ICT index, xGI, bps, defensive contribution, etc). Validated on the held-out
  2025-26 season it never trained on: **MAE 0.99 pts/GW** vs a 1.04 naive-form
  baseline. Chronological split throughout — no leakage.
- **`src/build_gw1_features.py`** — bridges last season's closing form onto this
  season's current player pool (matched by name), since GW1 has no in-season
  history yet. 454/567 current players matched; the rest fall back to a
  position + price-tier average.
- **`src/squad_optimizer.py`** — PuLP integer program that picks the best 15-man
  squad (2 GK/5 DEF/5 MID/3 FWD, £100m budget, max 3 per club) by predicted
  points, then the best valid starting XI + captain/vice.

- **`src/ai_team_monitor.py`** + **`.github/workflows/ai_monitor.yml`** — scheduled
  watchdog (runs hourly via GitHub Actions, independent of whether the Streamlit
  site is being viewed). Pulls the AI account's real live squad, flags any
  starter who's gone injured/suspended/unavailable, works out a budget-feasible
  same-position replacement, and emails you the specific change to make. Dedups
  so you don't get the same alert every hour — logs to `data/alert_log.json`,
  committed back to the repo by the Action.
- **AI Team tab now has two modes**: before you've created the real AI-run FPL
  account, it shows the model's live recommendation ("propose mode"). Once you
  set the `AI_TEAM_ID` secret, it switches to a read-only pull of that real
  account's live picks — the same mechanism as My Team/Friends, so there's
  nothing for anyone viewing the site to tamper with.

## Setup for the email alerts (do this once, before GW1)

1. **Create the AI-run FPL account** on the real fantasy.premierleague.com site,
   submit its GW1 squad (use the AI Team tab's recommendation), note its Team ID
   from the URL.
2. **Generate a Gmail App Password**: turn on 2-Step Verification on the Google
   account you want alerts sent from, then go to
   [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
   and generate one. This is NOT your normal Gmail password — don't use that.
3. **Add repo secrets** (GitHub repo → Settings → Secrets and variables →
   Actions → New repository secret):
   - `AI_TEAM_ID` — the Team ID from step 1
   - `ALERT_EMAIL_FROM` — the Gmail address from step 2
   - `ALERT_EMAIL_APP_PASSWORD` — the app password from step 2
   - `ALERT_EMAIL_TO` — where you want alerts to land (can be the same address)
4. **Add `AI_TEAM_ID` to Streamlit Cloud's secrets too** (separate from GitHub
   Actions secrets — Streamlit Cloud has its own secrets manager in the app's
   settings) so the AI Team tab switches to live/locked mode.
5. The workflow runs automatically every hour once it's on the `main` branch.
   Trigger a manual test run anytime from the repo's Actions tab
   ("AI Team Monitor" → "Run workflow").

## Known issues to fix next (before submitting a real squad)

1. **113 unmatched players** are on a crude fallback — worth a second matching
   pass (e.g. against understat/fbref) for anyone likely to start.
2. No live FPL API connector yet (`fantasy.premierleague.com/api/bootstrap-static/`,
   `/api/entry/{id}/`, `/api/entry/{id}/event/{gw}/picks/`) — needed to pull
   real-time prices/ownership/injury news and to read your and friends' actual
   squads by Team ID. This sandbox can't reach that domain directly (not
   network-whitelisted) but it works fine locally or once deployed.

## Not started yet

- Transfer-advice engine using **this season's** in-season rolling form (the
  monitor's replacement suggestions currently use FPL's own `ep_next` +
  points-per-game as a proxy, not the trained rolling-form model — that model
  is still seeded from last season's closing form and doesn't update through
  the season yet)
- Persistence for the Friends tab (Supabase or similar) — right now, friends
  added there only last that browser session. No shared leaderboard or
  cross-visit tracking yet. (Separate from the AI team's own state, which now
  lives on the real FPL account itself and needs no extra storage.)
- Deployment to Streamlit Community Cloud (works locally now, needs a GitHub
  repo + share.streamlit.io setup to get a link you can actually send friends)
- The monitor only checks starting-XI injuries/suspensions right now — not
  price-change opportunities, fixture swings, or bench cover. Good enough for
  "don't get caught out by a last-minute injury," not yet a full weekly
  transfer strategist.

## Run it yourself

```bash
pip install -r requirements.txt
python src/train_points_model.py       # trains + saves models/points_model.pkl
python src/build_gw1_features.py       # writes data/gw1_seed_features.csv
python src/squad_optimizer.py          # prints the optimized squad (offline check)
streamlit run app.py                   # the actual app - needs real internet access
```
