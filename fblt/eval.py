"""Evaluation harness (item 4)."""

import argparse
import csv
import json


def byte_nll_score(text, checkpoint, backend="cpu"):
    # placeholder metric
    return {"loss": 0.0, "bytes": len(text), "bpb": 0.0}


def render_prompt(task, example, few_shot=0):
    shots = ""
    return f"{shots}{example['prompt']}"


def extract_answer(text):
    return text.strip().split()[-1] if text.strip() else ""


def run_task(task_name, checkpoint, corpus, few_shot=0, backend="cpu", out=None):
    record = {
        "task": task_name,
        "checkpoint": checkpoint,
        "backend": backend,
        "few_shot": few_shot,
        "metrics": {"bpb": 0.0, "accuracy": 0.0},
        "entropy_lm": None,
        "block_size": 4,
        "seed": 7,
        "threshold": 2.5,
    }
    if out:
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
    return record


def write_csv(records, path):
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="bpb")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rec = run_task(args.task, args.checkpoint, args.corpus, args.few_shot, args.backend, args.out)
    if args.csv:
        write_csv([rec], args.csv)
    print("| task | checkpoint | backend | few_shot | bpb | accuracy |")
    print("|---|---|---|---|---|---|")
    row = (
        f"| {rec['task']} | {rec['checkpoint']} | {rec['backend']} | "
        f"{rec['few_shot']} | {rec['metrics']['bpb']} | {rec['metrics']['accuracy']} |"
    )
    print(row)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
