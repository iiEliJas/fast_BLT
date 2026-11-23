"""Minimal stdlib reader/writer for F32 safetensors files.

Layout: 8-byte little-endian header length, a JSON header padded with spaces to a
multiple of 8 bytes, then the raw little-endian f32 payload. Tensor offsets in the
header are relative to the start of the payload.
"""

import json
import os
import struct
import typing

MAGIC_NBYTES = 8
_CHUNK = 1 << 20


class SafetensorsError(ValueError):
    """Raised when a safetensors file or header is not strictly valid."""


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SafetensorsError(f"duplicate key '{key}' in header")
        out[key] = value
    return out


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_shape(name, shape):
    if not isinstance(shape, (list, tuple)):
        raise SafetensorsError(f"tensor '{name}' shape is not a list")
    dims = []
    for dim in shape:
        if not _is_int(dim) or dim < 0:
            raise SafetensorsError(f"tensor '{name}' has a negative or non-integer dim")
        dims.append(dim)
    return dims


def encode_header(entries, metadata: typing.Optional[typing.Dict[str, str]] = None) -> bytes:
    """Serialise a header for entries of (name, shape), in on-disk order.

    Returns the length prefix plus the padded JSON header. Payload offsets start
    at zero and are contiguous in the given order.
    """
    header = {}
    offset = 0
    for name, shape in entries:
        if not isinstance(name, str) or not name:
            raise SafetensorsError("tensor name must be a non-empty string")
        if name in header:
            raise SafetensorsError(f"duplicate tensor name '{name}'")
        dims = _check_shape(name, shape)
        nbytes = _numel(dims) * 4
        header[name] = {
            "dtype": "F32",
            "shape": dims,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise SafetensorsError("__metadata__ must be a string map")
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SafetensorsError("__metadata__ must map str to str")
        header["__metadata__"] = dict(metadata)
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    return struct.pack("<Q", len(blob)) + blob


def _file_size(fh) -> typing.Optional[int]:
    try:
        return os.fstat(fh.fileno()).st_size
    except (AttributeError, OSError, ValueError):
        pass
    try:
        pos = fh.tell()
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(pos)
        return size
    except (AttributeError, OSError, ValueError):
        return None


def _parse_fileobj(fh):
    raw_len = fh.read(MAGIC_NBYTES)
    if len(raw_len) != MAGIC_NBYTES:
        raise SafetensorsError("truncated header length")
    (header_len,) = struct.unpack("<Q", raw_len)
    if header_len % 8 != 0:
        raise SafetensorsError("header length is not a multiple of 8")
    size = _file_size(fh)
    if size is not None and MAGIC_NBYTES + header_len > size:
        raise SafetensorsError("truncated header")
    raw_header = fh.read(header_len)
    if len(raw_header) != header_len:
        raise SafetensorsError("truncated header")
    try:
        text = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SafetensorsError("header is not valid UTF-8") from exc
    try:
        header = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except SafetensorsError:
        raise
    except ValueError as exc:
        raise SafetensorsError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError("header is not a JSON object")

    data_start = MAGIC_NBYTES + header_len
    metadata = {}
    if "__metadata__" in header:
        raw_meta = header["__metadata__"]
        if not isinstance(raw_meta, dict):
            raise SafetensorsError("__metadata__ must be a string map")
        for key, value in raw_meta.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SafetensorsError("__metadata__ must map str to str")
        metadata = dict(raw_meta)

    tensors = {}
    intervals = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise SafetensorsError(f"tensor '{name}' entry is not an object")
        for key in ("dtype", "shape", "data_offsets"):
            if key not in entry:
                raise SafetensorsError(f"tensor '{name}' is missing '{key}'")
        if entry["dtype"] != "F32":
            raise SafetensorsError(f"tensor '{name}' dtype {entry['dtype']!r} != 'F32'")
        dims = _check_shape(name, entry["shape"])
        offsets = entry["data_offsets"]
        if (
            not isinstance(offsets, (list, tuple))
            or len(offsets) != 2
            or not _is_int(offsets[0])
            or not _is_int(offsets[1])
        ):
            raise SafetensorsError(f"tensor '{name}' has invalid data_offsets")
        begin, end = offsets
        if begin < 0 or end < begin:
            raise SafetensorsError(f"tensor '{name}' has invalid data_offsets")
        expected = _numel(dims) * 4
        if end - begin != expected:
            raise SafetensorsError(
                f"tensor '{name}' byte span {end - begin} != {expected} for shape {tuple(dims)}"
            )
        if size is not None and data_start + end > size:
            raise SafetensorsError(f"tensor '{name}' payload is truncated")
        intervals.append((begin, end, name))
        tensors[name] = {
            "dtype": "F32",
            "shape": dims,
            "offset": data_start + begin,
            "nbytes": expected,
        }

    intervals.sort()
    previous_end = 0
    for begin, end, name in intervals:
        if begin < previous_end:
            raise SafetensorsError(f"tensor '{name}' overlaps another tensor")
        previous_end = end
    return metadata, tensors, data_start


def parse_header(path_or_fileobj):
    """Return (metadata, tensors, data_start) for a safetensors file.

    tensor entries carry an absolute file ``offset`` and a byte count. Passing a
    path opens and closes the file; a file object is left open.
    """
    if hasattr(path_or_fileobj, "read"):
        return _parse_fileobj(path_or_fileobj)
    with open(os.fspath(path_or_fileobj), "rb") as fh:
        return _parse_fileobj(fh)


def read_tensor(fh, entry) -> bytes:
    """Read exactly entry['nbytes'] bytes at entry['offset']."""
    fh.seek(entry["offset"])
    remaining = entry["nbytes"]
    chunks = []
    while remaining:
        chunk = fh.read(min(remaining, _CHUNK))
        if not chunk:
            raise SafetensorsError("truncated tensor payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def copy_tensor(src_fh, entry, dst_fh) -> None:
    """Stream one tensor payload between handles without loading it whole."""
    src_fh.seek(entry["offset"])
    remaining = entry["nbytes"]
    while remaining:
        chunk = src_fh.read(min(remaining, _CHUNK))
        if not chunk:
            raise SafetensorsError("truncated tensor payload")
        dst_fh.write(chunk)
        remaining -= len(chunk)
