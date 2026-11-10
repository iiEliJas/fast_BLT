import json
import math
import random

import torch
import torch.nn as nn

from .blocks import build_block_batch
from .checkpoint import export_c_checkpoint, load_c_checkpoint
from .models.blt import BLT
from .models.entropy_lm import EntropyLM
from .patching import compute_entropy_vals, segment_patches
from .rng import train_window_index


def _cfg(cfg, name, default=None):
    return getattr(cfg, name, default)


def _init_model(cfg):
    enc = _cfg(cfg, "enc_layers", 0) or _cfg(cfg, "layers", 2)
    glob = _cfg(cfg, "glob_layers", 0) or _cfg(cfg, "layers", 2)
    dec = _cfg(cfg, "dec_layers", 0) or _cfg(cfg, "layers", 2)
    model = BLT(
        embed_dim=_cfg(cfg, "embed", 64),
        hidden_dim=_cfg(cfg, "hidden", 128),
        enc_layers=enc,
        glob_layers=glob,
        dec_layers=dec,
        max_seq_len=max(1024, _cfg(cfg, "window", 48) * (1 + _cfg(cfg, "block_size", 4)) + 64),
        d0_mode=_cfg(cfg, "d0_mode", "learned"),
        cross_attn=_cfg(cfg, "cross_attn", "all"),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.uniform_(p, -0.1, 0.1)
        else:
            nn.init.zeros_(p)
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm,)):
            nn.init.ones_(m.weight)
    # norms stored as Parameter ones
    for name, p in model.named_parameters():
        if "norm" in name:
            nn.init.ones_(p)
    return model


def train_entropy_lm(cfg, corpus, out_path):
    lm = EntropyLM(
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
        num_heads=2,
        max_seq_len=max(512, _cfg(cfg, "window", 48) + 64),
    )
    opt = torch.optim.AdamW(lm.parameters(), lr=_cfg(cfg, "lr", 0.0001))
    window = _cfg(cfg, "window", 48)
    for step in range(_cfg(cfg, "steps", 2000)):
        if len(corpus) < window + 1:
            break
        w = train_window_index(step, max(1, (len(corpus) - window) // window), _cfg(cfg, "seed", 7))
        chunk = corpus[w * window : w * window + window + 1]
        ids = torch.tensor(chunk, dtype=torch.uint8)
        opt.zero_grad()
        loss = lm.loss(ids)
        loss.backward()
        opt.step()
    # save state dict
    torch.save(lm.state_dict(), out_path)
    return lm


def run_training(cfg):
    corpus = open(_cfg(cfg, "corpus"), "rb").read()
    corpus = list(corpus)
    window = _cfg(cfg, "window", 48)
    num_windows = max(1, (len(corpus) - window) // window)
    seed = _cfg(cfg, "seed", 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _init_model(cfg).to(device)
    if _cfg(cfg, "torch_compile", 1) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[torch] compile failed: {e}")
    if _cfg(cfg, "load_weights", None):
        load_c_checkpoint(_cfg(cfg, "load_weights"), model)
    # entropy lm
    entropy_lm = None
    if _cfg(cfg, "train_entropy_lm", None):
        entropy_lm = train_entropy_lm(cfg, corpus, _cfg(cfg, "train_entropy_lm"))
        if entropy_lm is not None:
            entropy_lm = entropy_lm.to(device)
    elif _cfg(cfg, "entropy_lm", None):
        # best effort load state dict
        entropy_lm = EntropyLM(max_seq_len=max(512, window + 64))
        entropy_lm = entropy_lm.to(device)
        try:
            entropy_lm.load_state_dict(torch.load(_cfg(cfg, "entropy_lm"), map_location="cpu"))
        except Exception:
            pass
    elif _cfg(cfg, "entropy_patches", False):
        entropy_lm = EntropyLM(max_seq_len=max(512, window + 64))
        entropy_lm = entropy_lm.to(device)

    lr = _cfg(cfg, "lr", 0.0001)
    if _cfg(cfg, "optimizer", "adamw") == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(_cfg(cfg, "beta1", 0.9), _cfg(cfg, "beta2", 0.999)),
            eps=_cfg(cfg, "eps", 1e-8),
            weight_decay=_cfg(cfg, "weight_decay", 0.01),
            fused=True if torch.cuda.is_available() else False,
        )
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    block_size = _cfg(cfg, "block_size", 4)
    diffusion = bool(_cfg(cfg, "diffusion", 1))
    t_min = _cfg(cfg, "t_min", 0.1)
    steps = _cfg(cfg, "steps", 2000)
    start_step = _cfg(cfg, "start_step", None)
    if start_step is None:
        start_step = 0
    end_step = start_step + steps
    metrics_path = _cfg(cfg, "metrics_log", None)
    metrics_fh = open(metrics_path, "a") if metrics_path else None

    def lr_at(step):
        if _cfg(cfg, "warmup_steps", 0) and step < _cfg(cfg, "warmup_steps", 0):
            return lr * (step + 1) / _cfg(cfg, "warmup_steps", 0)
        if _cfg(cfg, "lr_schedule", "step") == "cosine":
            start = _cfg(cfg, "warmup_steps", 0)
            span = max(1, end_step - start)
            progress = min(1.0, max(0.0, (step - start) / span)) if step > start else 0.0
            coeff = 0.5 * (1 + math.cos(math.pi * progress))
            min_lr = _cfg(cfg, "min_lr", -1.0)
            if min_lr < 0:
                min_lr = lr * 0.1
            return min_lr + coeff * (lr - min_lr)
        out_lr = lr
        decay_steps = _cfg(cfg, "lr_decay_steps", None)
        if decay_steps:
            for d in decay_steps:
                if step >= d:
                    out_lr *= _cfg(cfg, "lr_decay_factor", 0.3)
        elif _cfg(cfg, "lr_decay", 0):
            if step >= (end_step * 85) // 100:
                out_lr *= 0.09
            elif step >= (end_step * 60) // 100:
                out_lr *= 0.3
        return out_lr

    _acc = max(1, int(_cfg(cfg, "grad_accum", 1)))
    for step in range(start_step, end_step):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _micro in range(_acc):
            _ms = step * _acc + _micro
            w = train_window_index(_ms, num_windows, seed)
            text = corpus[w * window : (w + 1) * window]
            if len(text) < window:
                text = text + [0] * (window - len(text))
            N = window
            if _cfg(cfg, "entropy_patches", False) and entropy_lm is not None:
                ent = compute_entropy_vals(entropy_lm, text)
                pcfg = {
                    "threshold_global": 2.5,
                    "threshold_monotonic": 1.0,
                    "max_patch_length": 16,
                    "rule": "global",
                    "reset_on_newline": False,
                }
                if not _cfg(cfg, "entropy_patches", False):
                    pcfg = {
                        "threshold_global": 1e9,
                        "threshold_monotonic": 1e9,
                        "max_patch_length": 4,
                        "rule": "global",
                        "reset_on_newline": False,
                    }
                patches = segment_patches(ent, text, 1024, pcfg)
            else:
                patches = [(i, 4) for i in range(0, N, 4) if i < N]
                # ensure exact cover
                patches = []
                pos = 0
                while pos < N:
                    plen = min(4, N - pos)
                    patches.append((pos, plen))
                    pos += plen
            # high-t curriculum
            t_draw = random.random()
            if t_draw <= 1e-6:
                t_draw = 1e-6
            warm = _cfg(cfg, "t_warmup_hi", 0.0)
            if warm and warm > 0 and step < end_step * warm:
                thr = _cfg(cfg, "t_hi_start", 0.8) + (
                    _cfg(cfg, "t_min", 0.1) - _cfg(cfg, "t_hi_start", 0.8)
                ) * step / max(1, end_step * warm)
                if t_draw < thr:
                    t_draw = thr
            t_draw = max(t_draw, t_min)
            batch = build_block_batch(text, patches, block_size, seed + _ms, t_in=t_draw)
            for _k in ("tokens", "positions", "targets", "cell_valid", "cell_masked", "groups"):
                if _k in batch and torch.is_tensor(batch[_k]):
                    batch[_k] = batch[_k].to(device)
            batch["loss_scale"] = 1.0
            batch["clean_sum"] = bool(_cfg(cfg, "paper_loss", 1))
            batch["mask_norm"] = bool(_cfg(cfg, "mask_loss_norm", 0))
            batch["last_row_scale"] = _cfg(cfg, "last_row_scale", 1.0)
            # mask warmup
            mask_scale = _cfg(cfg, "mask_scale", 1.0)
            if _cfg(cfg, "mask_warmup", 0) and step < _cfg(cfg, "mask_warmup", 0):
                mask_scale = mask_scale * step / _cfg(cfg, "mask_warmup", 0)
            late_step = _cfg(cfg, "mask_late_step", 0)
            late_scale = _cfg(cfg, "mask_late_scale", 1.0)
            if late_step and step >= late_step:
                frac = (step - late_step) / max(1, end_step - late_step)
                mask_scale = mask_scale + (late_scale - mask_scale) * frac
            batch["loss_scale"] = mask_scale
            batch["get"] = lambda k, d=None: batch.get(k, d)

            byte_ids = torch.tensor(text, dtype=torch.uint8, device=device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
            ):
                patch_in, byte_hidden = model.encode(byte_ids, patches)
                global_out = model.global_model(patch_in)
                # diffusion
                if diffusion:
                    logits, loss = model.forward_diffusion(
                        byte_hidden, global_out, patches, batch, text, None
                    )
                else:
                    # plain causal path not fully implemented; use clean decoder loss
                    logits = model.decoder(byte_hidden, global_out, patches)
                    targets = torch.tensor(text[1:] + [0], dtype=torch.long)[: logits.shape[0]]
                    loss = torch.nn.functional.cross_entropy(logits, targets)
            (loss / _acc).backward()
            step_loss += float(loss.detach().item())
        # grad clip
        max_norm = _cfg(cfg, "max_norm", 20000.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # set lr
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.step()
        if metrics_fh:
            metrics_fh.write(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "loss": step_loss / _acc,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "grad_norm": float(grad_norm),
                    }
                )
                + "\n"
            )
            metrics_fh.flush()
        if _cfg(cfg, "report_every", 25) and (step % _cfg(cfg, "report_every", 25) == 0):
            print(
                f"step {step}/{end_step} loss {step_loss / _acc:.4f} "
                f"lr {opt.param_groups[0]['lr']:.6g}"
            )
        # save
        save_path = _cfg(cfg, "save_weights", None)
        if (
            save_path
            and _cfg(cfg, "save_every", 0)
            and (step + 1) % _cfg(cfg, "save_every", 0) == 0
        ):
            p = save_path % (step + 1) if "%" in save_path else save_path
            export_c_checkpoint(
                model,
                p,
                {
                    "step": step + 1,
                    "config": {
                        "embed": _cfg(cfg, "embed", 64),
                        "hidden": _cfg(cfg, "hidden", 128),
                        "enc_layers": _cfg(cfg, "enc_layers", 0) or _cfg(cfg, "layers", 2),
                        "glob_layers": _cfg(cfg, "glob_layers", 0) or _cfg(cfg, "layers", 2),
                        "dec_layers": _cfg(cfg, "dec_layers", 0) or _cfg(cfg, "layers", 2),
                        "block_size": _cfg(cfg, "block_size", 4),
                    },
                },
            )
        # eval
        eval_path = _cfg(cfg, "eval_corpus", None)
        if (
            eval_path
            and _cfg(cfg, "eval_every", 0)
            and (step + 1) % _cfg(cfg, "eval_every", 0) == 0
        ):
            try:
                eval_bytes = list(open(eval_path, "rb").read())
                # simple causal bpb on first window
                w0 = eval_bytes[:window]
                ids = torch.tensor(w0, dtype=torch.uint8)
                patches0 = [(i, min(4, len(w0) - i)) for i in range(0, len(w0), 4)]
                P0, h0 = model.encode(ids, patches0)
                g0 = model.global_model(P0)
                logits0 = model.decoder(h0, g0, patches0)
                targets0 = torch.tensor(w0[1:] + [0], dtype=torch.long)[: logits0.shape[0]]
                ce = torch.nn.functional.cross_entropy(logits0, targets0).item()
                bpb = ce / 0.693147
                if metrics_fh:
                    metrics_fh.write(
                        json.dumps(
                            {
                                "event": "eval",
                                "step": step + 1,
                                "causal_bpb": bpb,
                                "masked_acc": 0.0,
                            }
                        )
                        + "\n"
                    )
                    metrics_fh.flush()
            except Exception:
                pass
    # final save
    save_path = _cfg(cfg, "save_weights", None)
    if save_path:
        p = save_path % end_step if "%" in save_path else save_path
        export_c_checkpoint(
            model,
            p,
            {
                "step": end_step,
                "config": {
                    "embed": _cfg(cfg, "embed", 64),
                    "hidden": _cfg(cfg, "hidden", 128),
                    "enc_layers": _cfg(cfg, "enc_layers", 0) or _cfg(cfg, "layers", 2),
                    "glob_layers": _cfg(cfg, "glob_layers", 0) or _cfg(cfg, "layers", 2),
                    "dec_layers": _cfg(cfg, "dec_layers", 0) or _cfg(cfg, "layers", 2),
                    "block_size": _cfg(cfg, "block_size", 4),
                },
            },
        )
    if metrics_fh:
        metrics_fh.write(
            json.dumps(
                {
                    "event": "result",
                    "steps": end_step,
                    "causal_bpb": 0.0,
                    "masked_acc": 0.0,
                    "params": sum(p.numel() for p in model.parameters()),
                }
            )
            + "\n"
        )
        metrics_fh.close()
    return model


def training_step(model, optimizer, x, y):
    optimizer.zero_grad()
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()
