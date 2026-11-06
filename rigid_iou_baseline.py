"""Stupid ShadowBench baseline: keep every finger at rest, wiggle two rigid hands for IoU.

Uses only the target mask and actual shadow mask. No CLIP, animal labels, book page,
joint search, or official score during optimization. This is an offline search:
it exceeds the episode's look budget and validates final poses statically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from shadowbench.factory import load_task, tasks_for
from shadowbench.native_environment import NativeGate, NativeRig, meshes_for
from shadowbench.pose import InvalidPose, Pose
from shadowbench.render import comparison, shadow, wall
from shadowbench.score import grade
from shadowbench.wrist_posture import posture_effort

START = Pose(np.zeros(46), np.array([[.1, -.12, 1.3], [-.1, -.12, 1.3]]),
             np.array([[1., 0, 0, 0], [1., 0, 0, 0]]))


def iou(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.count_nonzero(actual & target) /
                 max(1, np.count_nonzero(actual | target)))


def mutate(pose: Pose, rng: np.random.Generator, scale: float) -> Pose:
    candidate = pose.copy()
    hand = int(rng.integers(2))
    if rng.random() < .7:
        axis = int(rng.integers(3))
        candidate.positions[hand, axis] += float(rng.normal(0, scale * (1.4 if axis == 2 else 1.)))
    else:
        axis = np.zeros(3)
        axis[int(rng.integers(3))] = float(rng.choice([-1., 1.]))
        angle = float(rng.normal(0, scale * 180))
        delta = Rotation.from_rotvec(axis * np.radians(angle))
        root = Rotation.from_quat(candidate.quaternions[hand], scalar_first=True)
        candidate.quaternions[hand] = (delta * root).as_quat(scalar_first=True)
    return candidate


def save_gif(path: Path, task, history: list[tuple[int, float, np.ndarray]], max_frames: int) -> int:
    chosen = np.unique(np.linspace(0, len(history) - 1, min(max_frames, len(history)), dtype=int))
    frames = []
    font = ImageFont.load_default(size=14)
    target = wall(task.mask).resize((256, 192))
    for index in chosen:
        step, overlap, mask = history[int(index)]
        official = grade(mask, task, 1.)["score"]
        frame = Image.new("RGB", (768, 226), (246, 240, 225))
        frame.paste(target, (0, 34))
        frame.paste(wall(mask).resize((256, 192)), (256, 34))
        frame.paste(comparison(mask, task.mask).resize((256, 192)), (512, 34))
        ImageDraw.Draw(frame).text((8, 8), f"trial {step} | IoU {overlap:.3f} | official {official:.1f} (post hoc)",
                                   fill=(26, 31, 38), font=font)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[220] * (len(frames) - 1) + [1200],
                   loop=0, optimize=True)
    return len(frames)


def run(task_id: str, attempts: int, seed: int, out: Path, gif: bool, max_gif_frames: int) -> dict:
    task = load_task(task_id)
    rig = NativeRig()
    gate = NativeGate(rig)
    rng = np.random.default_rng(seed + sum(map(ord, task_id)))
    current = START.copy()
    gate.check(current)
    start_mask = shadow(meshes_for(rig, current))
    start_iou = iou(start_mask, task.mask)
    current_iou = start_iou
    best = current.copy()
    best_mask = start_mask
    best_iou = start_iou
    history = [(0, start_iou, start_mask.copy())] if gif else []
    accepted = 0
    valid = 0
    began = time.monotonic()
    for step in range(attempts):
        scale = (.08, .04, .02, .12)[(step // 100) % 4]
        candidate = mutate(current, rng, scale)
        try:
            meshes = meshes_for(rig, candidate)
            gate.check(candidate, meshes)
            mask = shadow(meshes)
        except (ValueError, InvalidPose):
            continue
        valid += 1
        candidate_iou = iou(mask, task.mask)
        if candidate_iou > best_iou:
            best, best_mask, best_iou = candidate.copy(), mask, candidate_iou
            if gif:
                history.append((step + 1, best_iou, best_mask.copy()))
        if candidate_iou >= current_iou:
            current = candidate
            current_iou = candidate_iou
            accepted += 1
        if (step + 1) % 150 == 0:
            current = best.copy()
            current_iou = best_iou
    gate.check(best)
    start_grade = grade(start_mask, task, posture_effort(START)["ease"])
    final_grade = grade(best_mask, task, posture_effort(best)["ease"])
    result = {
        "task_id": task_id, "revision": task.metadata["revision"],
        "start_iou": start_iou, "best_iou": best_iou,
        "start_score": start_grade["score"], "best_score": final_grade["score"],
        "official_components": final_grade, "attempts": attempts, "valid_candidates": valid,
        "accepted_moves": accepted, "elapsed_seconds": round(time.monotonic() - began, 2),
        "pose": best.dump(), "all_finger_joints_zero": bool(np.all(best.q == 0)),
        "protocol": "offline IoU search; static final-pose validity only",
    }
    dest = out / task_id
    dest.mkdir(parents=True, exist_ok=True)
    if gif:
        result["gif_frames"] = save_gif(dest / "trajectory.gif", task, history, max_gif_frames)
        result["gif_scope"] = "best-IoU improvements, uniformly sampled; official labels computed after optimization"
    (dest / "result.json").write_text(json.dumps(result, indent=2))
    wall(task.mask).save(dest / "target.png")
    wall(best_mask).save(dest / "best.png")
    comparison(best_mask, task.mask).save(dest / "comparison.png")
    print(json.dumps({key: result[key] for key in
                      ("task_id", "start_iou", "best_iou", "start_score", "best_score", "valid_candidates", "elapsed_seconds")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="repeatable; default all certified targets")
    parser.add_argument("--attempts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gif", action="store_true", help="save target/shadow/overlay progress GIF")
    parser.add_argument("--max-gif-frames", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "rigid_iou")
    args = parser.parse_args()
    task_ids = args.task or [entry["task_id"] for entry in tasks_for()]
    results = [run(task, args.attempts, args.seed, args.out, args.gif, args.max_gif_frames) for task in task_ids]
    print(json.dumps({"tasks": len(results),
                      "mean_start_score": float(np.mean([r["start_score"] for r in results])),
                      "mean_final_score": float(np.mean([r["best_score"] for r in results]))}), flush=True)


if __name__ == "__main__":
    main()
