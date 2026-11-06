"""Summarize and make a visual contact sheet for simple hand-shadow baselines."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    rows = []
    for path in sorted(root.glob("*/result.json")):
        data = json.loads(path.read_text())
        rows.append({
            "task_id": data["task_id"],
            "score": data.get("best_score", data.get("official_score")),
            "start_score": data.get("start_score", ""),
            "start_iou": data.get("start_iou", data.get("start_iou_from_initial_image", "")),
            "end_iou": data.get("best_iou", data.get("final_iou_from_look_images", "")),
            "attempts": data.get("attempts", data.get("probes", "")),
            "calls": data.get("calls", ""),
            "looks": data.get("look_images", ""),
            "all_finger_joints_zero": data["all_finger_joints_zero"],
        })
    if not rows:
        raise SystemExit("No results found")
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = [float(r["score"]) for r in rows]
    summary = {"tasks": len(rows), "mean_score": statistics.mean(values),
               "median_score": statistics.median(values),
               "min_score": min(values), "max_score": max(values),
               "all_finger_joints_zero": all(r["all_finger_joints_zero"] for r in rows),
               "max_calls": max((int(r["calls"]) for r in rows if r["calls"] != ""), default=None),
               "max_looks": max((int(r["looks"]) for r in rows if r["looks"] != ""), default=None)}
    (root / "summary.json").write_text(json.dumps({**summary, "rows": rows}, indent=2))
    cards = []
    sheet = Image.new("RGB", (5 * 512, ((len(rows) + 4) // 5) * 230), "#f3f0e9")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, row in enumerate(rows):
        task = row["task_id"]
        image_name = "best.png" if (root / task / "best.png").exists() else "final.png"
        target = Image.open(root / task / "target.png").convert("RGB")
        result = Image.open(root / task / image_name).convert("RGB")
        overlay = Image.open(root / task / "comparison.png").convert("RGB")
        x, y = (index % 5) * 512, (index // 5) * 230
        sheet.paste(target.resize((250, 188)), (x, y + 35))
        sheet.paste(result.resize((250, 188)), (x + 256, y + 35))
        draw.text((x + 4, y + 5), f"{task}  score {row['score']:.1f}", fill="#1c1f26", font=font)
        t = html.escape(task)
        cards.append(f'''<article><h2>{t} · {float(row["score"]):.2f}</h2><div class="images">
        <figure><img src="{t}/target.png"><figcaption>Target</figcaption></figure>
        <figure><img src="{t}/{image_name}"><figcaption>Rigid hands</figcaption></figure>
        <figure><img src="{t}/comparison.png"><figcaption>Overlay</figcaption></figure></div></article>''')
    sheet.save(root / "contact.png")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Rigid hands IoU baseline</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 16px;background:#f3f0e9;color:#1c1f26}}
article{{padding:20px;margin:22px 0;background:#fff;border-radius:12px}}
.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}}img{{width:100%;height:auto}}figcaption{{font-size:13px;color:#555}}
@media(max-width:700px){{.images{{display:block}}}}</style>
<h1>Rigid hands IoU baseline</h1><p>{len(rows)} certified targets. Mean official score {summary["mean_score"]:.2f}.
All finger joints stayed at their starting angles: {summary["all_finger_joints_zero"]}.</p>{''.join(cards)}</html>'''
    (root / "report.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
