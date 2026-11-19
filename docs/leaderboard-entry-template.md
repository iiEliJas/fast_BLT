# Leaderboard entry

Copy this template into your PR description and fill every field. The row itself comes from
`summary.json`; do not retype numbers from a log.

## Run

| | |
| --- | --- |
| run id | `<summary.run_id>` |
| summary file | `leaderboard/<run_id>.json` |
| status | `ok` / `unverified` / `synthetic` |
| hardware | `<hardware.cpu>` + `<hardware.gpu[0].name or "no GPU">` |
| backend | `cpu` / `cuda` |
| git commit | `<git_commit>` (`git_dirty: <true/false>`) |

## Command

```bash
# paste the exact command used, including --config and every override
bash scripts/speedrun.sh --backend cuda --out runs/<run_id>
```

## Config

```yaml
# paste config_path, or the effective config object from summary.json
```

## Results

| metric | value |
| --- | --- |
| causal BPB | `<metrics.causal_bpb>` |
| masked BPB | `<metrics.masked_bpb>` |
| masked acc | `<metrics.masked_acc>` |
| model params | `<model.params>` |
| train bytes | `<data.train.bytes>` |
| steps | `<training.steps>` |
| steps/s | `<throughput.steps_per_second>` |

## Checklist

- [ ] `git status --porcelain` was empty before the run (`git_dirty: false`), or the entry is
      marked `synthetic` on purpose.
- [ ] `python -m fblt.leaderboard --results <out>` reports `ok` for this run.
- [ ] The config is included, not just the architecture summary.
- [ ] The command is reproducible from the commit above on the stated hardware.
- [ ] `<out>/summary.json` is added under `leaderboard/<run_id>.json` in this PR.
- [ ] Nothing in the record was edited by hand; if it was, say what and why.
