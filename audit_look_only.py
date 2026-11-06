"""Check the saved tool logs against the published call/look budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED = {"reset", "plan", "move_hand", "turn_hand", "look", "undo", "submit"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    reports = []
    for directory in sorted(args.run_dir.iterdir()):
        if not directory.is_dir() or not (directory / "result.json").is_file():
            continue
        result = json.loads((directory / "result.json").read_text())
        records = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        assert records[0]["tool"] == "reset"
        assert records[1]["tool"] == "plan"
        assert records[-1]["tool"] == "submit"
        assert all(record["tool"] in ALLOWED for record in records)
        assert [record["i"] for record in records] == list(range(len(records)))
        assert all(all(value == 0 for value in record["after"]["q"]) for record in records)
        calls = len(records) - 1
        looks = sum(record["tool"] == "look" and bool(record["image"]) for record in records)
        assert calls == result["calls"] <= 100
        assert looks == result["look_images"] <= 32
        assert abs(records[-1]["metrics"]["score"] - result["official_score"]) < 1e-9
        assert all((directory / record["image"]).is_file()
                   for record in records if record["image"])
        reports.append({"task_id": directory.name, "calls": calls, "looks": looks,
                        "score": result["official_score"], "all_joints_zero": True})
    if not reports:
        raise ValueError("No episodes found")
    audit = {"episodes": len(reports), "max_calls": max(x["calls"] for x in reports),
             "max_looks": max(x["looks"] for x in reports),
             "tools_seen": sorted(ALLOWED), "all_logged_joint_angles_zero": True,
             "checks": "plan first; allowed tools only; within call/look budgets; images present; final score consistent",
             "note": "Policy's lack of hidden-reward access is established by reviewing look_only_baseline.py, not by log checks alone.",
             "per_task": reports}
    (args.run_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: audit[key] for key in ("episodes", "max_calls", "max_looks")}, indent=2))


if __name__ == "__main__":
    main()
