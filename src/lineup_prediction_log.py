"""
lineup_prediction_log.py — accuracy tracking for the EARLY lineup signal,
going forward, not backward.

NEXT_STEPS.md has the full story: fpledits.com (and every other source
checked) only exposes its LATEST predicted-lineup snapshot, not a
historical archive, so there's no way to honestly backtest this the way
squad_optimizer.py/risk_analyzer.py's fixes were backtested against a full
held-out season. Per instructions, rather than ship an unvalidated
accuracy claim, this logs a snapshot before each gameweek and scores it
against actual starts once that gameweek is finished - a real, growing,
honest track record instead of a one-time historical number.

Usage:
  log_snapshot(gw)   - call this (e.g. from ai_team_monitor.py's hourly
                        run) to record "as of now, who's predicted to
                        start GW{gw}". Overwrites any earlier snapshot for
                        the same gameweek, so the log naturally ends up
                        holding whatever was predicted closest to kickoff.
  score_finished_gameweek(gw) - call this once a gameweek is over, to
                        compare that snapshot against who actually started
                        (via each squad's real per-GW history) and append
                        the result to the running accuracy log.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fpl_api
import lineup_predictor

BASE = Path(__file__).resolve().parent.parent
SNAPSHOT_LOG_PATH = BASE / "data" / "lineup_prediction_snapshots.json"
ACCURACY_LOG_PATH = BASE / "data" / "lineup_prediction_accuracy.json"


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save(path: Path, data: dict):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def log_snapshot(gw: int) -> bool:
    """Record the current predicted-starting-XI set for gameweek `gw`.
    Returns False (and logs nothing) if the source was unreachable -
    never overwrite a real snapshot with an empty one just because this
    particular check happened to fail."""
    predicted_ids = lineup_predictor.fetch_predicted_starting_ids()
    if predicted_ids is None:
        return False

    log = _load(SNAPSHOT_LOG_PATH)
    log[str(gw)] = {
        "predicted_starting_ids": sorted(predicted_ids),
        "logged_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    _save(SNAPSHOT_LOG_PATH, log)
    return True


def score_finished_gameweek(gw: int) -> dict | None:
    """Compares gameweek `gw`'s logged snapshot (if any) against who
    actually started, via fpl_api.get_event_live() - every player's real
    gameweek stats in a single request, not one call per player. Returns
    None if there's no snapshot to score, or the gameweek hasn't finished
    (live/ returns no elements until kickoff).

    'Actually started' = minutes > 0 that gameweek - a started-and-
    immediately-subbed-off player still counts as correctly predicted,
    since the signal is about starting, not full 90s."""
    snapshots = _load(SNAPSHOT_LOG_PATH)
    snap = snapshots.get(str(gw))
    if snap is None:
        return None

    live = fpl_api.get_event_live(gw)
    if live is None or not live.get("elements"):
        return None  # gameweek hasn't been played (or FPL hasn't published stats) yet

    predicted_ids = set(snap["predicted_starting_ids"])
    correct = 0
    total = 0
    for row in live["elements"]:
        pid = row["id"]
        actually_started = row["stats"]["minutes"] > 0
        predicted_started = pid in predicted_ids
        total += 1
        if actually_started == predicted_started:
            correct += 1

    if total == 0:
        return None

    result = {
        "gameweek": gw,
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4),
        "scored_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    accuracy_log = _load(ACCURACY_LOG_PATH)
    accuracy_log[str(gw)] = result
    _save(ACCURACY_LOG_PATH, accuracy_log)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["log", "score"])
    parser.add_argument("gameweek", type=int)
    args = parser.parse_args()

    if args.action == "log":
        ok = log_snapshot(args.gameweek)
        print(f"Snapshot logged for GW{args.gameweek}." if ok
              else f"Could not reach the early-prediction source - nothing logged for GW{args.gameweek}.")
    else:
        result = score_finished_gameweek(args.gameweek)
        print(result if result else f"No snapshot on file for GW{args.gameweek}, or nothing scorable yet.")
