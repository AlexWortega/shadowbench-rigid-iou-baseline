"""Fetch the small, public ShadowBench source and certified targets from HF."""
import json
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
REPO = "https://huggingface.co/datasets/perpetual-research/shadowbench"
REVISION = "795a17b86e4f24095c0551849909c32482cfe183"
with urllib.request.urlopen(f"https://huggingface.co/api/datasets/perpetual-research/shadowbench/tree/{REVISION}?recursive=true") as response:
    entries = json.load(response)
for entry in entries:
    path = entry["path"]
    wanted = (
        path.startswith("shadowbench/")
        or path.startswith("scripts/")
        or path.startswith("assets/ultraleap/")
        or path.startswith("data/bench/targets/")
        or (path.startswith("data/bench/books/") and path.endswith(("/target.png", "/task.json", "/source.png", "/target.svg")))
        or path == "data/bench/books/index.json"
        or path == "data/tasks/tasks.parquet"
        or path in ("README.md", "pyproject.toml", "LICENSE.md")
    )
    if entry.get("type") != "file" or not wanted:
        continue
    destination = ROOT / path
    if destination.exists():
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO}/resolve/{REVISION}/{urllib.parse.quote(path)}"
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        output.write(response.read())
    print(path, destination.stat().st_size, flush=True)
