"""Dataset download and cache helpers."""

import hashlib
import os
import shutil
import urllib.request

CACHE_DIR = os.path.expanduser("~/.cache/fblt/datasets")


def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def download(url, dest, sha256=None):
    tmp = dest + ".tmp"
    if os.path.exists(url):
        shutil.copyfile(url, tmp)
    else:
        urllib.request.urlretrieve(url, tmp)
    if sha256:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != sha256:
            os.remove(tmp)
            raise ValueError("checksum mismatch")
    os.replace(tmp, dest)
    return dest


def ensure_dataset(name, url, sha256=None):
    p = cache_path(name)
    if not os.path.exists(p):
        download(url, p, sha256)
    return p
