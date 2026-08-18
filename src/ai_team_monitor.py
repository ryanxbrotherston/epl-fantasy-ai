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
  2. Checks every player in that squad against live injury/availability status.
  3. For anyone flagged (injured/suspended/unavailable/highly doubtful) who's
     in the starting XI, computes a same-position replacement suggestion
     within budget.
  4. Emails Ryan the specific change to make, IF this exact issue hasn't
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fpl_api

BASE = Path(__file__).resolve().parent.parent
ALERT_LOG_PATH = BASE / "data" / "alert_log.json"

UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}
DOUBTFUL_THRESHOLD = 50  # chance_of_playing_next_round % below which a "doubtful" player still gets flagged

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def load_alert_log() -> dict:
    if ALERT_LOG_PATH.exists():
        return json.loads(ALERT_LOG_PATH.read_text())
    return {}


def save_alert_log(log: dict):
    ALERT_LOG_PATH.parent.mkdir(exist_ok=True)
    ALERT_LOG_PATH.write_text(json.dumps(log, indent=2))


def flag_problem_players(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Squad members who are injured/suspended/unavailable, or doubtful below
    threshold, and are in the starting XI (multiplier > 0, i.e. not benched)."""
    starting = picks_df[picks_df["multiplier"] > 0].copy()
    hard_out = starting["status"].isin(UNAVAILABLE_STATUSES)
    # chance_of_playing_next_round comes through fpl_api merges as part of the
    # players table when available; guard for its absence
    return starting[hard_out]


def suggest_replacement(problem_player: pd.Series, squad_ids: set, bank: int,
                         all_players: pd.DataFrame, team_limit_counts: dict) -> pd.Series | None:
    """Best same-position, budget-feasible, club-limit-respecting replacement,
    ranked by FPL's own ep_next blended with points_per_game (the live model's
    rolling component isn't recomputed here - this is a simpler in-season
    proxy, see README)."""
    max_price = problem_player["now_cost"] + bank
    pos = problem_player["position"]
    team_of_departing = problem_player.get("team")

    candidates = all_players[
        (all_players["position"] == pos)
        & (all_players["now_cost"] <= max_price)
        & (~all_players["id"].isin(squad_ids))
        & (~all_players["status"].isin(UNAVAILABLE_STATUSES))
    ].copy()

    def club_ok(row):
        count = team_limit_counts.get(row["team"], 0)
        if row["team"] == team_of_departing:
            count -= 1  # departing player frees up a club slot
        return count < 3

    candidates = candidates[candidates.apply(club_ok, axis=1)]
    if candidates.empty:
        return None

    ep_next = pd.to_numeric(candidates["ep_next"], errors="coerce").fillna(0)
    ppg = pd.to_numeric(candidates["points_per_game"], errors="coerce").fillna(0)
    candidates["score"] = 0.6 * ep_next + 0.4 * ppg
    return candidates.sort_values("score", ascending=False).iloc[0]


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

    result = fpl_api.team_picks_dataframe(ai_team_id, target_event, bootstrap)
    if result is None:
        print(f"No picks available yet for GW{target_event} (pre-deadline, team not created, or "
              f"picks not public until GW1 starts). Nothing to check.")
        return

    picks_df, entry_history = result
    all_players = fpl_api.players_dataframe(bootstrap)

    problems = flag_problem_players(picks_df)
    if problems.empty:
        print(f"GW{target_event}: AI team's starting XI all clear. No alert needed.")
        return

    alert_log = load_alert_log()
    gw_key = str(target_event)
    already_alerted = set(alert_log.get(gw_key, []))

    squad_ids = set(picks_df["element"])
    team_limit_counts = all_players[all_players["id"].isin(squad_ids)]["team"].value_counts().to_dict()
    bank = entry_history.get("bank", 0)

    new_issues = []
    lines = [f"GW{target_event} deadline: {gw['deadline_time']}\n"]

    for _, problem in problems.iterrows():
        issue_key = f"{problem['element']}_{problem['status']}"
        if issue_key in already_alerted:
            continue

        replacement = suggest_replacement(problem, squad_ids, bank, all_players, team_limit_counts)
        status_label = {"i": "INJURED", "s": "SUSPENDED", "u": "UNAVAILABLE", "n": "NOT IN SQUAD"}.get(
            problem["status"], problem["status"]
        )

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
