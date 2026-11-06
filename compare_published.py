"""Paired, descriptive comparison to the public ShadowBench task metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_run", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/tasks.parquet"))
    args = parser.parse_args()
    published = {r["task_id"]: r for r in pq.read_table(
        args.tasks, columns=["task_id", "revision", "rest_score", "blob_score",
                             "search_score", "search_evaluations", "best_score"]
    ).to_pylist()}
    rows = []
    for path in sorted(args.offline_run.glob("*/result.json")):
        ours = json.loads(path.read_text())
        theirs = published[ours["task_id"]]
        if ours["revision"] != theirs["revision"]:
            raise ValueError(f"Target revision differs for {ours['task_id']}")
        rows.append({
            "task_id": ours["task_id"],
            "ours": ours["best_score"],
            "ours_evaluations": ours["attempts"] + 1,
            "rest": theirs["rest_score"],
            "target_convex_hull": theirs["blob_score"],
            "blind_cma_search": theirs["search_score"],
            "blind_cma_evaluations": theirs["search_evaluations"],
            "best_model": theirs["best_score"],
            "ours_minus_cma": ours["best_score"] - theirs["search_score"],
        })
    if not rows:
        raise ValueError("No result.json files")
    with (args.offline_run / "published_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    difference = np.array([r["ours_minus_cma"] for r in rows])
    rng = np.random.default_rng(20261106)
    means = rng.choice(difference, size=(20_000, len(difference)), replace=True).mean(axis=1)
    summary = {
        "targets": len(rows),
        "mean_ours": float(np.mean([r["ours"] for r in rows])),
        "mean_rest": float(np.mean([r["rest"] for r in rows])),
        "mean_target_convex_hull": float(np.mean([r["target_convex_hull"] for r in rows])),
        "mean_blind_cma": float(np.mean([r["blind_cma_search"] for r in rows])),
        "mean_blind_cma_evaluations": float(np.mean([r["blind_cma_evaluations"] for r in rows])),
        "mean_best_model": float(np.mean([r["best_model"] for r in rows])),
        "beats_cma": int(np.sum(difference > 0)),
        "beats_best_model": sum(r["ours"] > r["best_model"] for r in rows),
        "paired_mean_ours_minus_cma": float(np.mean(difference)),
        "paired_bootstrap_95_percent_interval": [float(x) for x in np.quantile(means, [.025, .975])],
        "comparison_scope": "offline static-pose search versus published task metadata; not a protocol-matched agent leaderboard comparison",
    }
    (args.offline_run / "published_comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
