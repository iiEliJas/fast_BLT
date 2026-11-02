import argparse
import dataclasses
import typing

import yaml as _yaml

T = typing.TypeVar("T")


def _field_to_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _to_argv(cfg: typing.Any) -> typing.List[str]:
    argv: typing.List[str] = []
    defaults = {f.name: f.default for f in dataclasses.fields(cfg)}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        default = defaults[f.name]
        if f.name in ("corpus", "backend"):
            argv.extend([_field_to_flag(f.name), str(val)])
            continue
        if f.name == "threshold":
            unmask = getattr(cfg, "unmask", None)
            if unmask == "eb" and val is None:
                argv.extend(["--threshold", "1.0"])
                continue
        if val == default:
            continue
        tp = f.type
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        if origin is typing.Union and type(None) in args:
            if val is None:
                continue
            inner = [a for a in args if a is not type(None)]
            inner_tp = inner[0] if inner else None
            if typing.get_origin(inner_tp) is list:
                argv.extend([_field_to_flag(f.name), ",".join(str(v) for v in val)])
            else:
                argv.extend([_field_to_flag(f.name), str(val)])
        elif tp is bool or (isinstance(default, bool) and val is not None):
            if val:
                argv.append(_field_to_flag(f.name))
        elif origin is list:
            if val is None:
                continue
            argv.extend([_field_to_flag(f.name), ",".join(str(v) for v in val)])
        else:
            argv.extend([_field_to_flag(f.name), str(val)])
    return argv


@dataclasses.dataclass
class TrainConfig:
    corpus: str = ""

    steps: int = 2000
    lr: float = 0.0001
    warmup_steps: int = 0
    lr_schedule: str = "step"
    min_lr: float = -1.0
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-08
    weight_decay: float = 0.01
    weight_decay_norms: bool = False
    max_norm: float = 20000.0
    grad_accum: int = 1
    paper_loss: int = 1
    mask_loss_norm: int = 0
    seed: int = 7
    backend: str = "cpu"
    deterministic: bool = False
    bf16: bool = False
    fp8: bool = False
    flash_attn: bool = False

    embed: int = 64
    hidden: int = 128
    layers: int = 2
    enc_layers: int = 0
    glob_layers: int = 0
    dec_layers: int = 0
    cross_attn: str = "all"
    d0_mode: str = "learned"

    diffusion: int = 1
    t_min: float = 0.1
    t_warmup_hi: float = 0.0
    t_hi_start: float = 0.8
    block_size: int = 4
    window: int = 48

    mask_warmup: int = 0
    mask_scale: float = 1.0
    mask_late_step: int = 0
    mask_late_scale: float = 1.0
    last_row_scale: float = 1.0

    lr_decay: int = 0
    lr_decay_steps: typing.Optional[typing.List[int]] = None
    lr_decay_factor: float = 0.3

    entropy_patches: bool = False
    train_entropy_lm: typing.Optional[str] = None
    entropy_lm: typing.Optional[str] = None

    eval_corpus: typing.Optional[str] = None
    eval_windows: int = 200
    eval_skip: int = 0
    monitor_every: int = 2000
    monitor_windows: int = 24
    eval_every: int = 0

    save_weights: typing.Optional[str] = None
    save_dir: typing.Optional[str] = None
    resume: typing.Optional[str] = None
    save_optim: typing.Optional[str] = None
    save_every: int = 0
    save_best: typing.Optional[str] = None
    best_metric: str = "bpb"
    save_best_optim: typing.Optional[str] = None
    load_weights: typing.Optional[str] = None
    load_optim: typing.Optional[str] = None
    start_step: typing.Optional[int] = None
    report_every: int = 25
    cuda_scratch_mb: int = 0
    scratch_mb: int = 0
    model_mb: int = 0
    peak_tflops: float = 0.0
    peak_bandwidth: float = 0.0

    grad_norm_log: typing.Optional[str] = None
    update_norm_log: typing.Optional[str] = None
    component_norm_log: typing.Optional[str] = None
    activation_dump_log: typing.Optional[str] = None
    batch_log: typing.Optional[str] = None
    loss_log: typing.Optional[str] = None
    metrics_log: typing.Optional[str] = None

    def to_argv(self) -> typing.List[str]:
        return _to_argv(self)


@dataclasses.dataclass
class InferConfig:
    checkpoint: str = ""

    embed: int = 0
    hidden: int = 0
    enc_layers: int = 0
    glob_layers: int = 0
    dec_layers: int = 0
    backend: str = "cpu"

    prompt: typing.Optional[str] = None
    prompt_file: typing.Optional[str] = None

    new_bytes: int = 64
    num_samples: int = 1
    method: str = "greedy"
    seed: int = 11

    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    repeat_penalty: float = 1.0
    no_repeat_ngram: int = 0
    max_seq_len: int = 0

    k: int = 8
    certify_positions: bool = False

    block_size: int = 4
    unmask: str = "confidence"
    threshold: typing.Optional[float] = None
    boundary_aligned: bool = False
    adaptive: bool = False
    b_min: int = 4
    b_max: int = 16
    accept_target: float = 0.5
    adapt_window: int = 8

    cross_attn: str = "all"

    fixed_patches: bool = False
    patch_threshold_global: float = 3.0
    patch_threshold_monotonic: float = 1.0
    max_patch_length: int = 16
    entropy_lm: typing.Optional[str] = None
    allow_random_entropy_lm: bool = False

    output: typing.Optional[str] = None

    def to_argv(self) -> typing.List[str]:
        return _to_argv(self)


def _hyphen_to_underscore(key: str) -> str:
    return key.replace("-", "_")


def load_config(
    config_class: typing.Any,
    yaml_path: typing.Optional[str] = None,
    overrides: typing.Optional[typing.List[str]] = None,
) -> typing.Any:
    defaults = {f.name: f.default for f in dataclasses.fields(config_class)}

    kw: typing.Dict[str, typing.Any] = {}
    if yaml_path is not None:
        with open(yaml_path, "r") as fh:
            raw = _yaml.safe_load(fh) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")

        for raw_key, val in raw.items():
            py_name = _hyphen_to_underscore(raw_key)
            if py_name not in defaults:
                raise ValueError(f"Unknown config key: {raw_key}")
            field = next(f for f in dataclasses.fields(config_class) if f.name == py_name)
            kw[py_name] = _coerce(val, field.type, defaults[py_name])

    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Override must be key=value, got: {ov}")
            raw_key, raw_val = ov.split("=", 1)
            py_name = _hyphen_to_underscore(raw_key)
            if py_name not in defaults:
                raise ValueError(f"Unknown config key in override: {raw_key}")
            field = next(f for f in dataclasses.fields(config_class) if f.name == py_name)
            kw[py_name] = _coerce(raw_val, field.type, defaults[py_name])

    return config_class(**kw)


def _coerce(val: typing.Any, tp: typing.Any, default: typing.Any) -> typing.Any:
    if val is None:
        return None

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if typing.get_origin(tp) is typing.Union and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if inner:
            return _coerce(val, inner[0], default)
        return val

    if tp is int:
        return int(val) if not isinstance(val, int) else val
    if tp is float:
        return float(val) if not isinstance(val, (int, float)) else val
    if tp is bool:
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")
    if tp is str:
        return str(val) if not isinstance(val, str) else val
    if origin is list:
        if isinstance(val, list):
            return [_coerce(v, args[0], None) for v in val]
        return [_coerce(v.strip(), args[0], None) for v in str(val).split(",")]

    return val


def _field_to_flag_pub(name: str) -> str:
    return "--" + name.replace("_", "-")


def _cli_type(field: dataclasses.Field) -> typing.Any:
    """argparse type for a dataclass field, unwrapping Optional[...]."""
    tp = field.type
    origin = typing.get_origin(tp)
    if origin is typing.Union and type(None) in typing.get_args(tp):
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        tp = inner[0] if inner else str
        origin = typing.get_origin(tp)
    if origin is list:
        return str
    if tp is bool:
        return None
    if tp in (int, float, str):
        return tp
    return str


def add_config_args(
    parser: argparse.ArgumentParser,
    config_class: typing.Any,
    skip: typing.Sequence[str] = (),
    overrides: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> None:
    """Give every dataclass field a real command-line flag.

    The wrapper used to require --override key=value for anything not listed in
    argparse by hand, which meant the C binary's full flag surface was reachable
    only through a stringly-typed escape hatch. Deriving the flags from the
    dataclass keeps the three in step: a field that exists is a flag you can pass,
    and its type comes from the annotation.

    Every generated flag defaults to None so "not given" stays distinguishable
    from "given the default value". That is what lets precedence work:
    dataclass default < YAML < --override < explicit flag.

    Values land in `overrides` instead of a namespace, so callers can apply them
    after load_config() without a second code path.
    """
    if overrides is None:
        overrides = {}
    skip_set = set(skip)
    for field in dataclasses.fields(config_class):
        if field.name in skip_set:
            continue
        flag = _field_to_flag_pub(field.name)
        default = field.default
        help_text = f"[config] {field.name}"
        if default is not None and default is not dataclasses.MISSING:
            help_text += f" (default: {default})"

        if field.type is bool or (isinstance(default, bool) and default is not False):
            parser.add_argument(
                flag,
                dest=field.name,
                action="store_true",
                default=None,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag,
                dest=field.name,
                type=_cli_type(field),
                default=None,
                help=help_text,
            )
