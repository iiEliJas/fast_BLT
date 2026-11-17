"""Dataset download and cache (item 14)."""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request

try:
    import fcntl
except ImportError:
    fcntl = None

RETRIES = 3


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str
    size: int = 0
    license: str = "unknown"
    source: str = ""
    version: str = ""


# Each entry must carry a working URL and the SHA-256 of the bytes that URL returns;
# download_dataset refuses to write a manifest otherwise. Checksums are reproducible:
# see docs/DATASETS.md for how they were recorded and how to refresh one.
REGISTRY = {
    "tinyshakespeare": DatasetSpec(
        name="tinyshakespeare",
        url="https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt",
        sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        size=1115394,
        license="public-domain",
        source="karpathy/char-rnn (Shakespeare)",
        version="char-rnn@master:input.txt",
    ),
    "tinystories": DatasetSpec(
        name="tinystories",
        url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStoriesV2-GPT4-valid.txt",
        sha256="6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
        size=22502601,
        license="CDLA-Sharing-1.0",
        source="roneneldan/TinyStories (TinyStoriesV2-GPT4-valid)",
        version="TinyStoriesV2-GPT4-valid.txt@main",
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


def get_spec(name):
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise ValueError(f"unknown dataset '{name}' (known: {known})") from None


def validate_spec(spec):
    if not spec.url:
        raise ValueError(f"dataset '{spec.name}' has no download URL in REGISTRY")
    if len(spec.sha256) != 64 or any(c not in "0123456789abcdef" for c in spec.sha256):
        raise ValueError(
            f"dataset '{spec.name}' has no valid SHA-256 in REGISTRY; "
            "refusing an unverified download"
        )


def download_one(url, dest, sha256=None, retries=RETRIES):
    tmp = dest + ".tmp"
    for attempt in range(retries):
        try:
            if os.path.exists(url):
                shutil.copyfile(url, tmp)
            else:
                urllib.request.urlretrieve(url, tmp)
            if sha256 and _sha256_file(tmp) != sha256:
                raise ValueError(f"checksum mismatch for {url}")
            os.replace(tmp, dest)
            return dest
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (2**attempt))


def acquire_lock(name):
    lock_path = os.path.join(dataset_dir(name), ".lock")
    fh = open(lock_path, "w")
    if fcntl is not None:
        fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def download_dataset(name):
    spec = get_spec(name)
    validate_spec(spec)
    d = dataset_dir(name)
    fh = acquire_lock(name)
    try:
        manifest_path = os.path.join(d, "manifest.json")
        if os.path.exists(manifest_path):
            return manifest_path  # already published
        shard_path = os.path.join(d, "shard-00000.bin")
        download_one(spec.url, shard_path, spec.sha256)
        size = os.path.getsize(shard_path)
        if spec.size and size != spec.size:
            os.remove(shard_path)
            raise ValueError(f"size mismatch for '{name}': expected {spec.size}, got {size}")
        manifest = {
            "name": name,
            "url": spec.url,
            "version": spec.version,
            "license": spec.license,
            "source": spec.source,
            "size": size,
            "sha256": spec.sha256,
            "shards": 1,
        }
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, manifest_path)
        return manifest_path
    finally:
        if fcntl is not None:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download and verify registered datasets.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", metavar="NAME")
    ap.add_argument("--verify", metavar="NAME")
    args = ap.parse_args(argv)
    if args.list:
        for name, spec in sorted(REGISTRY.items()):
            print(f"{name}: {spec.size or '?'} bytes, license={spec.license}, url={spec.url}")
        return 0
    if args.download:
        try:
            print("manifest:", download_dataset(args.download))
        except (KeyError, ValueError, OSError) as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.verify:
        ok = verify_dataset(args.verify)
        print("verify:", ok)
        return 0 if ok else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
