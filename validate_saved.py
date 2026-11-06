"""Independently rerender offline final poses and recheck score/static validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shadowbench.factory import load_task
from shadowbench.native_environment import NativeGate, NativeRig, meshes_for
from shadowbench.pose import Pose
from shadowbench.render import shadow
from shadowbench.score import grade
from shadowbench.wrist_posture import posture_effort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    rig = NativeRig()
    gate = NativeGate(rig)
    details = []
    for path in sorted(root.glob("*/result.json")):
        saved = json.loads(path.read_text())
        pose = Pose.parse(saved["pose"])
        assert np.all(pose.q == 0)
        meshes = meshes_for(rig, pose)
        gate.check(pose, meshes)
        mask = shadow(meshes)
        pixels = np.asarray(Image.open(path.parent / "best.png").convert("RGB"))
        saved_mask = np.all(pixels == [26, 31, 38], axis=-1)
        assert np.array_equal(mask, saved_mask), saved["task_id"]
        task = load_task(saved["task_id"])
        assert task.metadata["revision"] == saved["revision"]
        score = grade(mask, task, posture_effort(pose)["ease"])["score"]
        assert abs(score - saved["best_score"]) < 1e-5, saved["task_id"]
        details.append({"task_id": saved["task_id"], "score": score})
    if not details:
        raise ValueError("No saved results found")
    result = {"validated_targets": len(details), "all_finger_joints_zero": True,
              "all_saved_pixels_equal": True, "all_static_gates_passed": True,
              "all_official_scores_equal": True,
              "scope": "Final static poses only; episode budget and continuous motion path are not checked.",
              "per_target": details}
    (root / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("validated_targets", "all_static_gates_passed", "all_official_scores_equal")}, indent=2))


if __name__ == "__main__":
    main()
