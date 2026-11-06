"""Tiny tool-agent baseline: move rigid hands, measure IoU from its own look images.

No hidden reward, renderer calls, book hints, model, CLIP, or finger changes are
used by the policy. The recorded run uses a legal plan, <=100 calls and <=32
look images. Initial observation is free, as in the published agent harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import fftconvolve

from shadowbench.factory import create, tasks_for
from shadowbench.render import wall, comparison

INK = np.array([26, 31, 38], dtype=np.uint8)


def panels(image) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(image.convert("RGB"))
    target = np.all(pixels[:384, :512] == INK, axis=-1)
    actual = np.all(pixels[:384, 512:1024] == INK, axis=-1)
    return target, actual


def iou(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.count_nonzero(actual & target) /
                 max(1, np.count_nonzero(actual | target)))


def best_shift(actual: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    correlation = fftconvolve(target.astype(np.float32), actual[::-1, ::-1].astype(np.float32), mode="full")
    cy, cx = np.array(correlation.shape) // 2
    window = correlation[cy - 110:cy + 111, cx - 150:cx + 151]
    y, x = np.unravel_index(np.argmax(window), window.shape)
    return int(x - 150), int(y - 110)


def propose(rng: np.random.Generator, probe: int, target: np.ndarray,
            actual: np.ndarray) -> tuple[str, dict]:
    hand = ("right", "left")[probe % 2]
    if probe < 4:
        dx, dy = best_shift(actual, target)
        pixels_per_meter = 512 / .8 * 2.5 / (2.5 - 1.3)
        delta = np.array([dx / pixels_per_meter * .5, -dy / pixels_per_meter * .5, 0.])
        delta = np.clip(delta, -.10, .10)
        if np.linalg.norm(delta) < .008:
            delta[int(rng.integers(2))] = float(rng.choice([-.04, .04]))
        return "move_hand", {"hand": hand, "delta": delta.tolist()}
    if probe in (4, 5):
        scale = .08 if target.sum() > actual.sum() else -.08
        return "move_hand", {"hand": hand, "delta": [0., 0., scale]}
    if rng.random() < .72:
        axis = int(rng.integers(3))
        delta = [0., 0., 0.]
        delta[axis] = float(rng.choice([-1., 1.]) * rng.choice([.035, .07, .10]))
        return "move_hand", {"hand": hand, "delta": delta}
    axis = [0., 0., 0.]
    axis[int(rng.integers(3))] = 1.
    return "turn_hand", {"hand": hand, "axis": axis,
                         "degrees": float(rng.choice([-20., -10., 10., 20.]))}


def run(task_id: str, seed: int, out: Path) -> dict:
    dest = out / task_id
    env = create(task_id, dest)
    began = time.monotonic()
    initial = env.observe()
    initial.save(dest / "initial.png")
    target, actual = panels(initial)
    current_iou = iou(actual, target)
    start_iou = current_iou
    rng = np.random.default_rng(seed + sum(map(ord, task_id)))
    env.call("plan", {"subject": "shadow silhouette",
                      "arrangement": "Leave all fingers at rest; move and turn the two whole hands to overlap the target."})
    accepted = refused = undone = probes = 0
    while not env.done and env.images < 32 and env.calls <= 96:
        name, args = propose(rng, probes, target, actual)
        probes += 1
        response = env.call(name, args)
        if not response.text.startswith("Accepted"):
            refused += 1
            continue
        observed = env.call("look", {"view": "front"})
        if observed.image is None:
            raise RuntimeError("accepted movement did not yield look image")
        _, new_actual = panels(observed.image)
        score = iou(new_actual, target)
        if score > current_iou:
            accepted += 1
            current_iou, actual = score, new_actual
        else:
            back = env.call("undo", {})
            if not back.text.startswith("Accepted"):
                raise RuntimeError(f"undo failed: {back.text}")
            undone += 1
    if not env.done:
        env.call("submit", {})
    result = {
        "task_id": task_id, "start_iou_from_initial_image": start_iou,
        "final_iou_from_look_images": current_iou,
        "official_score": env.final["score"], "official_components": env.final,
        "calls": env.calls, "look_images": env.images, "probes": probes,
        "accepted_moves": accepted, "refused_moves": refused, "undone_moves": undone,
        "all_finger_joints_zero": bool(np.all(env.pose.q == 0)),
        "elapsed_seconds": round(time.monotonic() - began, 2),
        "policy_observations": "initial image, look images and tool acceptance/refusal text only",
    }
    (dest / "result.json").write_text(json.dumps(result, indent=2))
    wall(target).save(dest / "target.png")
    wall(actual).save(dest / "final.png")
    comparison(actual, target).save(dest / "comparison.png")
    print(json.dumps({key: result[key] for key in
                      ("task_id", "official_score", "calls", "look_images", "accepted_moves", "refused_moves", "undone_moves", "elapsed_seconds")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "look_only")
    args = parser.parse_args()
    tasks = args.task or [x["task_id"] for x in tasks_for()]
    results = [run(task, args.seed, args.out) for task in tasks]
    print(json.dumps({"tasks": len(results), "mean_official_score": float(np.mean([x["official_score"] for x in results]))}), flush=True)


if __name__ == "__main__":
    main()
