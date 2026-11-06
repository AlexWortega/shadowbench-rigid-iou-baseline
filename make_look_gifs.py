"""Animate the images actually returned to the look-only agent, with post hoc scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = np.array([26, 31, 38], dtype=np.uint8)


def visible_iou(image: Image.Image) -> float:
    pixels = np.asarray(image.convert("RGB"))
    target = np.all(pixels[:384, :512] == INK, axis=-1)
    actual = np.all(pixels[:384, 512:1024] == INK, axis=-1)
    return float(np.count_nonzero(target & actual) /
                 max(1, np.count_nonzero(target | actual)))


def frame(image: Image.Image, title: str) -> Image.Image:
    source = image.convert("RGB")
    result = Image.new("RGB", (768, 178), (246, 240, 225))
    locations = ((0, 0), (512, 0), (0, 384), (512, 384))
    for index, (x, y) in enumerate(locations):
        panel = source.crop((x, y, x + 512, y + 384)).resize((192, 144))
        result.paste(panel, (index * 192, 34))
    ImageDraw.Draw(result).text((8, 7), title, fill=(26, 31, 38),
                                font=ImageFont.load_default(size=13))
    return result


def run(root: Path) -> None:
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "result.json").exists():
            continue
        records = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        looks = [record for record in records if record["tool"] == "look" and record["image"]]
        initial = Image.open(directory / "initial.png")
        frames = [frame(initial, f"initial | visible IoU {visible_iou(initial):.3f} | official {records[0]['metrics']['score']:.1f} (post hoc)")]
        for count, record in enumerate(looks, 1):
            image = Image.open(directory / record["image"])
            frames.append(frame(image, f"call {record['i']} | look {count}/{len(looks)} | IoU {visible_iou(image):.3f} | official {record['metrics']['score']:.1f} (post hoc)"))
        frames[0].save(directory / "trajectory.gif", save_all=True, append_images=frames[1:],
                       duration=[220] * (len(frames) - 1) + [1200], loop=0, optimize=True)
        print(directory.name, len(frames), (directory / "trajectory.gif").stat().st_size, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()
