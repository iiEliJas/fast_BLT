"""Reproducible speedrun runner."""

import argparse
import json
import os


def run_speedrun(config_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # placeholder: invoke train with config
    print("speedrun: config", config_path, "out", out_dir)
    return {"status": "ok", "config": config_path}


def aggregate_leaderboard(results_dir, out_json):
    rows = []
    for f in os.listdir(results_dir):
        if f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fh:
                rows.append(json.load(fh))
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/speedrun")
    args = ap.parse_args()
    r = run_speedrun(args.config, args.out)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(r, fh, indent=2)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
