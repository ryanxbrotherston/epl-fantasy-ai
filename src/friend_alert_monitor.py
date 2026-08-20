"""
friend_alert_monitor.py — Scheduled watchdog for friends' OWN FPL squads,
opt-in only.

Runs on a schedule (see .github/workflows/friend_monitor.yml), same reason
ai_team_monitor.py does: Streamlit Community Cloud apps only execute when
someone's browsing them, so the actual scheduled checking has to live in
GitHub Actions instead.

This is a sibling to ai_team_monitor.py, not a merge with it - see
DECISION_friend_alerts.md for why the two stay separate scripts (isolation:
a bug or outage here must never take down Ryan's own AI-team alerting) while
sharing the actual injury/suspension/confirmed-lineup checking logic via
src/squad_alert_checks.py.

What it does each run:
  1. Reads every manager_profiles row (Supabase) where email_alerts_enabled
     is true and fpl_team_id is set - the opt-in checkbox in app.py's My
     Team tab is the only way a row gets into this set. See
     DECISION_friend_alerts.md for why this is a dedicated consent flag,
     never implied by just having a saved Team ID.
  2. For each opted-in manager, pulls THEIR OWN current squad from the live
     FPL API and runs the exact same checks ai_team_monitor.py runs against
     the AI team - official FPL status, official confirmed team sheet
     (highlightly.net, within the kickoff window), and the early
     bench-likely signal - via squad_alert_checks.py, so any fix to those
     checks benefits both monitors automatically.
  3. Emails each flagged manager individually, at THEIR OWN saved email (not
     Ryan), personalized with their display_name if present. Dedup is a
     separate log, data/friend_alert_log.json, keyed by
     {gameweek: {team_id: [issue_keys]}} - deliberately not shared with
     ai_team_monitor.py's data/alert_log.json (see DECISION_friend_alerts.md).

Config comes from environment variables (set as GitHub Actions secrets -
never hardcode these):
  SUPABASE_URL              - same Supabase project app.py's manager_store.py uses
  SUPABASE_SECRET_KEY       - Supabase secret key (bypasses RLS - see manager_store.py)
  ALERT_EMAIL_FROM          - Gmail address to send from (same sender as the AI monitor)
  ALERT_EMAIL_APP_PASSWORD  - Gmail App Password for that address
"""

import json
import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fpl_api
import lineup_predictor
import manager_store
from squad_alert_checks import (
    flag_problem_players,
    flag_bench_likely_players,
    flag_confirmed_benched_players,
    suggest_replacement,
)

BASE = Path(__file__).resolve().parent.parent
FRIEND_ALERT_LOG_PATH = BASE / "data" / "friend_alert_log.json"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def load_friend_alert_log() -> dict:
    if FRIEND_ALERT_LOG_PATH.exists():
        return json.loads(FRIEND_ALERT_LOG_PATH.read_text())
    return {}


def save_friend_alert_log(log: dict):
    FRIEND_ALERT_LOG_PATH.parent.mkdir(exist_ok=True)
    FRIEND_ALERT_LOG_PATH.write_text(json.dumps(log, indent=2))


def opted_in_managers() -> list[dict]:
    """Every manager_profiles row with the consent checkbox on AND a saved
    Team ID to check - both required, since a row can have one without the
    other (e.g. alerts toggled on before ever saving a Team ID)."""
    client = manager_store.get_client()
    resp = client.table(manager_store.TABLE).select("*").eq("email_alerts_enabled", True).execute()
    return [row for row in resp.data if row.get("fpl_team_id")]


def send_alert_email(to_addr: str, subject: str, body: str):
    from_addr = os.environ["ALERT_EMAIL_FROM"]
    app_password = os.environ["ALERT_EMAIL_APP_PASSWORD"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(from_addr, app_password)
        server.sendmail(from_addr, [to_addr], msg.as_string())


def check_manager(manager: dict, bootstrap: dict, target_event: int, gw: dict,
                   all_players: pd.DataFrame, predicted_starting_ids: set | None,
                   already_alerted: set) -> tuple[list[str], list[str]]:
    """Runs the shared checks for one manager's own squad. Returns
    (email_body_lines, new_issue_keys) - both empty if there's nothing new
    to tell them about."""
    team_id = manager["fpl_team_id"]
    result = fpl_api.team_picks_dataframe(team_id, target_event, bootstrap)
    if result is None:
        print(f"  Team {team_id}: no picks available yet for GW{target_event}. Skipping.")
        return [], []

    picks_df, entry_history = result

    problems = flag_problem_players(picks_df)
    confirmed_benched = flag_confirmed_benched_players(
        picks_df, all_players, bootstrap, target_event, set(problems["element"])
    )
    bench_likely = flag_bench_likely_players(
        picks_df, predicted_starting_ids,
        set(problems["element"]) | set(confirmed_benched["element"]),
    )

    if problems.empty and confirmed_benched.empty and bench_likely.empty:
        print(f"  Team {team_id}: starting XI all clear. No alert needed.")
        return [], []

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

    return lines, new_issues


def main():
    managers = opted_in_managers()
    if not managers:
        print("No managers currently opted in to email alerts. Nothing to check.")
        return

    bootstrap = fpl_api.get_bootstrap_static()
    gw = fpl_api.current_gameweek(bootstrap)
    target_event = gw["id"]
    all_players = fpl_api.players_dataframe(bootstrap)

    # Same early-signal fetch ai_team_monitor.py uses - no separate snapshot
    # log here, that ongoing accuracy tracking is ai_team_monitor.py's job
    # (it runs every hour regardless of whether the AI team has picks yet);
    # this script just needs the current predicted set for its own check.
    predicted_starting_ids = lineup_predictor.fetch_predicted_starting_ids()
    if predicted_starting_ids is None:
        print("Note: fpledits.com's early-prediction source was unreachable this run - "
              "skipping the early/bench-likely check for everyone this time.")

    alert_log = load_friend_alert_log()
    gw_key = str(target_event)
    gw_log = alert_log.get(gw_key, {})

    total_emails = 0
    for manager in managers:
        team_id = str(manager["fpl_team_id"])
        already_alerted = set(gw_log.get(team_id, []))

        print(f"Checking {manager.get('display_name') or manager.get('email') or team_id} "
              f"(team {team_id})...")
        lines, new_issues = check_manager(
            manager, bootstrap, target_event, gw, all_players, predicted_starting_ids, already_alerted,
        )
        if not new_issues:
            continue

        to_addr = manager.get("email")
        if not to_addr:
            print(f"  Team {team_id}: has new issues but no saved email on file. Skipping send.")
            continue

        greeting = f"Hi {manager['display_name']}," if manager.get("display_name") else "Hi,"
        body = (
            f"{greeting}\n\n" + "\n".join(lines) +
            "\n\nMake this change on your account before the deadline above."
        )
        send_alert_email(to_addr, f"⚠️ Your FPL Team GW{target_event}: action needed before deadline", body)
        print(f"  Sent alert email for {len(new_issues)} new issue(s).")
        total_emails += 1

        gw_log[team_id] = list(already_alerted | set(new_issues))

    alert_log[gw_key] = gw_log
    save_friend_alert_log(alert_log)
    print(f"Done. Sent {total_emails} alert email(s) across {len(managers)} opted-in manager(s).")


if __name__ == "__main__":
    main()
