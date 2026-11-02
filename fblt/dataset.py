"""Dataset download and cache (item 14)."""

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import time
import urllib.request


@dataclasses.dataclass
class DatasetSpec:
    name: str
    url: str
    sha256: str
    shards: int = 1
    license: str = "unknown"
    source: str = ""


REGISTRY = {
    "tinystories": DatasetSpec(
        name="tinystories",
        url="",
        sha256="",
        shards=1,
        license="CDLA-Sharing-1.0",
        source="TinyStories",
    ),
}


def cache_root():
    return os.environ.get("FBLT_CACHE", os.path.expanduser("~/.cache/fblt"))


def dataset_dir(name):
    d = os.path.join(cache_root(), "datasets", name)
    os.makedirs(d, exist_ok=True)
    return d


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(url, dest, sha256=None, retries=3):
    for attempt in range(retries):
        try:
            tmp = dest + ".tmp"
            if os.path.exists(url):
                shutil.copyfile(url, tmp)
            else:
                urllib.request.urlretrieve(url, tmp)
            if sha256:
                if _sha256_file(tmp) != sha256:
                    os.remove(tmp)
                    raise ValueError("checksum mismatch")
            os.replace(tmp, dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (2**attempt))


def acquire_lock(name):
    lock_path = os.path.join(dataset_dir(name), ".lock")
    fh = open(lock_path, "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def download_dataset(name):
    spec = REGISTRY[name]
    d = dataset_dir(name)
    fh = acquire_lock(name)
    try:
        manifest_path = os.path.join(d, "manifest.json")
        if os.path.exists(manifest_path):
            return manifest_path  # no-op
        # download shard(s)
        shard_path = os.path.join(d, "shard-00000.bin")
        if spec.url:
            download_one(spec.url, shard_path, spec.sha256 if spec.sha256 else None)
        size = os.path.getsize(shard_path) if os.path.exists(shard_path) else 0
        sha = _sha256_file(shard_path) if os.path.exists(shard_path) else ""
        manifest = {
            "name": name,
            "license": spec.license,
            "source": spec.source,
            "size": size,
            "sha256": sha,
            "shards": spec.shards,
        }
        # write manifest last
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, manifest_path)
        return manifest_path
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def verify_dataset(name):
    d = dataset_dir(name)
    manifest_path = os.path.join(d, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path) as f:
        m = json.load(f)
    shard = os.path.join(d, "shard-00000.bin")
    if not os.path.exists(shard):
        return False
    return _sha256_file(shard) == m["sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download")
    ap.add_argument("--verify")
    args = ap.parse_args()
    if args.list:
        for k, v in REGISTRY.items():
            print(f"{k}: license={v.license} shards={v.shards}")
        return
    if args.download:
        p = download_dataset(args.download)
        print("manifest:", p)
        return
    if args.verify:
        ok = verify_dataset(args.verify)
        print("verify:", ok)
        return


if __name__ == "__main__":
    main()
