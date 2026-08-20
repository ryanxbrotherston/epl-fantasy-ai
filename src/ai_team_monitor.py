"""
ai_team_monitor.py — Scheduled watchdog for the AI-run FPL team.

Runs on a schedule (see .github/workflows/ai_monitor.yml), NOT inside the
Streamlit app. Streamlit Community Cloud apps only execute when someone's
browsing them - they can't run a background job on their own - so this lives
in GitHub Actions instead, which can run on a cron schedule independent of
whether anyone's looking at the site.

What it does each run:
  1. Pulls the AI team's actual current squad from the live FPL API (its
     real Team ID - this is the source of truth, not anything we store).
  2. Checks every player in that squad against live injury/availability
     status - FPL's own OFFICIAL designation (injured/suspended/
     unavailable/highly doubtful), not a prediction.
  3. Separately checks the starting XI against confirmed_lineup.py's
     OFFICIAL team sheet (via highlightly.net, free tier - see
     NEXT_STEPS.md for the source-vetting trail, including the two
     candidates that didn't pan out first). Only ever checked within
     CONFIRMED_LOOKUP_WINDOW_HOURS of a player's own kickoff - lineups
     aren't released until ~30-40min before anyway, and the free tier
     (100 req/day) isn't enough to poll every fixture every hour all week.
  4. Also checks the starting XI against lineup_predictor.py's EARLY
     signal (fpledits.com's predicted-lineup snapshot) - anyone who looks
     bench-likely there but isn't already covered by a more authoritative
     check above gets flagged too, clearly labeled as an early/predicted
     warning, not an official one.
  5. For anyone flagged in any category who's in the starting XI, computes
     a same-position replacement suggestion within budget.
  6. Emails Ryan the specific change to make, IF this exact issue hasn't
     already been alerted this gameweek (dedup via data/alert_log.json,
     committed back to the repo by the Action after each run).

Config comes from environment variables (set as GitHub Actions secrets -
never hardcode these):
  AI_TEAM_ID              - the AI-run FPL account's Team ID
  ALERT_EMAIL_FROM         - Gmail address to send from
  ALERT_EMAIL_APP_PASSWORD - Gmail App Password (not your normal password -
                              generate one at myaccount.google.com/apppasswords,
                              needs 2FA enabled on the Google account first)
  ALERT_EMAIL_TO           - where the alert should land (can be same as FROM)
"""

import json
import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fpl_api
import lineup_predictor
import lineup_prediction_log
from squad_alert_checks import (
    flag_problem_players,
    flag_bench_likely_players,
    flag_confirmed_benched_players,
    suggest_replacement,
)

BASE = Path(__file__).resolve().parent.parent
ALERT_LOG_PATH = BASE / "data" / "alert_log.json"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def load_alert_log() -> dict:
    if ALERT_LOG_PATH.exists():
        return json.loads(ALERT_LOG_PATH.read_text())
    return {}


def save_alert_log(log: dict):
    ALERT_LOG_PATH.parent.mkdir(exist_ok=True)
    ALERT_LOG_PATH.write_text(json.dumps(log, indent=2))


def send_alert_email(subject: str, body: str):
    from_addr = os.environ["ALERT_EMAIL_FROM"]
    app_password = os.environ["ALERT_EMAIL_APP_PASSWORD"]
    to_addr = os.environ.get("ALERT_EMAIL_TO", from_addr)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(from_addr, app_password)
        server.sendmail(from_addr, [to_addr], msg.as_string())


def main():
    ai_team_id = int(os.environ["AI_TEAM_ID"])

    bootstrap = fpl_api.get_bootstrap_static()
    gw = fpl_api.current_gameweek(bootstrap)
    target_event = gw["id"]

    # Snapshot the early signal + score last gameweek's snapshot against
    # what actually happened, every hourly run, independent of whether the
    # AI team has picks yet - this is what builds the "log accuracy going
    # forward" track record over the season (see lineup_prediction_log.py -
    # no historical archive exists to backtest against, so this is done
    # forward instead, honestly, rather than shipping an unvalidated claim).
    predicted_starting_ids = lineup_predictor.fetch_predicted_starting_ids()
    if predicted_starting_ids is None:
        print("Note: fpledits.com's early-prediction source was unreachable this run - "
              "skipping the snapshot log and the early/bench-likely check this time.")
    else:
        lineup_prediction_log.log_snapshot(target_event)
    if target_event > 1:
        score = lineup_prediction_log.score_finished_gameweek(target_event - 1)
        if score:
            print(f"GW{target_event - 1} early-prediction accuracy: "
                  f"{score['correct']}/{score['total']} ({score['accuracy']:.1%})")

    result = fpl_api.team_picks_dataframe(ai_team_id, target_event, bootstrap)
    if result is None:
        print(f"No picks available yet for GW{target_event} (pre-deadline, team not created, or "
              f"picks not public until GW1 starts). Nothing to check.")
        return

    picks_df, entry_history = result
    all_players = fpl_api.players_dataframe(bootstrap)

    problems = flag_problem_players(picks_df)
    confirmed_benched = flag_confirmed_benched_players(
        picks_df, all_players, bootstrap, target_event, set(problems["element"])
    )
    bench_likely = flag_bench_likely_players(
        picks_df, predicted_starting_ids,
        set(problems["element"]) | set(confirmed_benched["element"]),
    )

    if problems.empty and confirmed_benched.empty and bench_likely.empty:
        print(f"GW{target_event}: AI team's starting XI all clear (official status, official "
              f"team sheet, and early-prediction checks). No alert needed.")
        return

    alert_log = load_alert_log()
    gw_key = str(target_event)
    already_alerted = set(alert_log.get(gw_key, []))

    squad_ids = set(picks_df["element"])
    team_limit_counts = all_players[all_players["id"].isin(squad_ids)]["team"].value_counts().to_dict()
    bank = entry_history.get("bank", 0)

    new_issues = []
    lines = [f"GW{target_event} deadline: {gw['deadline_time']}\n"]

    if not problems.empty:
        lines.append("=== CONFIRMED - FPL's own official status ===")
    for _, problem in problems.iterrows():
        issue_key = f"{problem['element']}_{problem['status']}"
        if issue_key in already_alerted:
            continue

        replacement = suggest_replacement(problem, squad_ids, bank, all_players, team_limit_counts)
        status_label = {
            "i": "INJURED", "s": "SUSPENDED", "u": "UNAVAILABLE", "n": "NOT IN SQUAD",
            "d": f"DOUBTFUL ({problem.get('chance_of_playing_next_round')}% chance)",
        }.get(problem["status"], problem["status"])

        if replacement is not None:
            lines.append(
                f"• {problem['web_name']} ({status_label}) — suggest transferring IN "
                f"{replacement['web_name']} (£{replacement['now_cost']/10:.1f}m, "
                f"{replacement['team_name']}) instead."
            )
        else:
            lines.append(
                f"• {problem['web_name']} ({status_label}) — no budget-feasible same-position "
                f"replacement found within your bank (£{bank/10:.1f}m). Manual look needed."
            )
        new_issues.append(issue_key)

    if not confirmed_benched.empty:
        lines.append("\n=== CONFIRMED - official team sheet (highlightly.net) ===")
    for _, benched_player in confirmed_benched.iterrows():
        issue_key = f"{benched_player['element']}_confirmed_benched"
        if issue_key in already_alerted:
            continue

        replacement = suggest_replacement(benched_player, squad_ids, bank, all_players, team_limit_counts)
        if replacement is not None:
            lines.append(
                f"• {benched_player['web_name']} — officially NOT in the confirmed starting XI. "
                f"Suggest transferring IN {replacement['web_name']} "
                f"(£{replacement['now_cost']/10:.1f}m, {replacement['team_name']}) instead."
            )
        else:
            lines.append(
                f"• {benched_player['web_name']} — officially NOT in the confirmed starting XI. "
                f"No budget-feasible same-position replacement found within your bank "
                f"(£{bank/10:.1f}m). Manual look needed."
            )
        new_issues.append(issue_key)

    if not bench_likely.empty:
        lines.append("\n=== EARLY WARNING - predicted only, from fpledits.com's predicted "
                      "lineups, NOT an official team sheet - could easily change before "
                      "kickoff ===")
    for _, bench_player in bench_likely.iterrows():
        issue_key = f"{bench_player['element']}_early_bench_likely"
        if issue_key in already_alerted:
            continue

        replacement = suggest_replacement(bench_player, squad_ids, bank, all_players, team_limit_counts)
        if replacement is not None:
            lines.append(
                f"• {bench_player['web_name']} (not in the predicted starting XI) — consider "
                f"transferring IN {replacement['web_name']} (£{replacement['now_cost']/10:.1f}m, "
                f"{replacement['team_name']}) - but this is an early prediction, not confirmed. "
                f"Worth a second check closer to the deadline before acting."
            )
        else:
            lines.append(
                f"• {bench_player['web_name']} (not in the predicted starting XI) — no "
                f"budget-feasible same-position replacement found within your bank "
                f"(£{bank/10:.1f}m). This is an early prediction, not confirmed."
            )
        new_issues.append(issue_key)

    if not new_issues:
        print(f"GW{target_event}: issues found but already alerted previously. Skipping duplicate email.")
        return

    body = "\n".join(lines) + "\n\nMake this change on the AI account before the deadline above."
    send_alert_email(f"⚠️ AI Team GW{target_event}: action needed before deadline", body)
    print(f"Sent alert email for {len(new_issues)} new issue(s).")

    alert_log[gw_key] = list(already_alerted | set(new_issues))
    save_alert_log(alert_log)


if __name__ == "__main__":
    main()
