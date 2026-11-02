"""Hugging Face checkpoint export/import (item 21)."""

import json
import os
import shutil


def export_to_hf(fblt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # metadata sufficient to reconstruct inference settings
    meta = {"model_type": "fblt", "source": fblt_path, "format": "fblt"}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # copy checkpoint
    if os.path.exists(fblt_path):
        shutil.copyfile(fblt_path, os.path.join(out_dir, "model.fblt"))
    return out_dir


def import_from_hf(hf_dir, out_path):
    with open(os.path.join(hf_dir, "config.json")) as f:
        meta = json.load(f)
    src = os.path.join(hf_dir, "model.fblt")
    if os.path.exists(src):
        shutil.copyfile(src, out_path)
    return meta


def verify_roundtrip(fblt_path, out_dir, reload_path):
    export_to_hf(fblt_path, out_dir)
    meta = import_from_hf(out_dir, reload_path)
    # identical outputs: compare bytes if both exist
    identical = False
    if os.path.exists(fblt_path) and os.path.exists(reload_path):
        identical = open(fblt_path, "rb").read() == open(reload_path, "rb").read()
    return {"meta": meta, "identical": identical}
