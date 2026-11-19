#!/usr/bin/env bash
# Run a small seed/config series and build a leaderboard over the results.
#
# Each iteration forwards the caller's pass-through flags first, then appends
# the generated --seed, --config and --out. argparse keeps the last occurrence,
# so a stray --out/--seed in the pass-through cannot collapse every run into one
# directory. Build flags are forwarded the same way speedrun.sh does it: build
# stays ON unless --no-build is given (the first iteration compiles, the rest
# are no-ops).
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/miniseries.sh [--seeds LIST] [--configs LIST] [--out DIR]
                             [--backend cpu|cuda] [--steps N]
                             [--build|--no-build] [--dry-run]

Runs seed x config speedruns, then renders a leaderboard over them.

  -h, --help      Show this help and exit.
  --seeds LIST    Comma-separated seeds (default: 1; must not be empty).
  --configs LIST  Comma-separated YAML configs (default: built-in config).
  --out DIR       Output root (default: runs/miniseries).
  --backend B     cpu or cuda (forwarded).
  --steps N       Training steps per run (forwarded).
  --build         Configure and compile before the first run (default).
  --no-build      Reuse binaries that are already built.
  --dry-run       Print the plan and every command; run and write nothing.

Generated --seed/--config/--out flags are always appended after the
pass-through flags, so they win under argparse's last-flag-wins rule.
EOF
}

SEEDS="1"
CONFIGS=""
OUT="runs/miniseries"
BACKEND=""
STEPS=""
DRY_RUN=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --seeds)
            [[ $# -ge 2 ]] || { echo "miniseries.sh: --seeds needs a value" >&2; exit 2; }
            SEEDS="$2"
            shift 2
            ;;
        --configs)
            [[ $# -ge 2 ]] || { echo "miniseries.sh: --configs needs a value" >&2; exit 2; }
            CONFIGS="$2"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "miniseries.sh: --out needs a value" >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        --backend)
            [[ $# -ge 2 ]] || { echo "miniseries.sh: --backend needs a value" >&2; exit 2; }
            BACKEND="$2"
            shift 2
            ;;
        --steps)
            [[ $# -ge 2 ]] || { echo "miniseries.sh: --steps needs a value" >&2; exit 2; }
            STEPS="$2"
            shift 2
            ;;
        --build|--no-build)
            PASSTHRU+=("$1")
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "miniseries.sh: unknown argument: $1" >&2
            echo "Try --help." >&2
            exit 2
            ;;
    esac
done

if [[ -z "$SEEDS" ]]; then
    echo "miniseries.sh: --seeds must not be empty" >&2
    exit 2
fi

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
if [[ -n "$CONFIGS" ]]; then
    IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
else
    CONFIG_LIST=("")
fi

run_count=$(( ${#SEED_LIST[@]} * ${#CONFIG_LIST[@]} ))
echo "plan: ${#SEED_LIST[@]} seed(s) x ${#CONFIG_LIST[@]} config(s) -> ${run_count} run(s) under ${OUT}"

for seed in "${SEED_LIST[@]}"; do
    if [[ -z "$seed" ]]; then
        echo "miniseries.sh: empty seed in --seeds" >&2
        exit 2
    fi
    for cfg in "${CONFIG_LIST[@]}"; do
        if [[ -n "$cfg" ]]; then
            stem="$(basename "$cfg")"
            stem="${stem%.*}"
            run_dir="${OUT}/seed${seed}-${stem}"
        else
            run_dir="${OUT}/seed${seed}"
        fi

        cmd=(python3 -m fblt.speedrun)
        if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
            cmd+=("${PASSTHRU[@]}")
        fi
        if [[ -n "$BACKEND" ]]; then
            cmd+=(--backend "$BACKEND")
        fi
        if [[ -n "$STEPS" ]]; then
            cmd+=(--steps "$STEPS")
        fi
        if [[ "$DRY_RUN" -eq 1 ]]; then
            cmd+=(--dry-run)
        fi
        cmd+=(--seed "$seed")
        if [[ -n "$cfg" ]]; then
            cmd+=(--config "$cfg")
        fi
        cmd+=(--out "$run_dir")

        printf '$'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        if [[ "$DRY_RUN" -eq 0 ]]; then
            "${cmd[@]}"
        fi
    done
done

lb_cmd=(python3 -m fblt.leaderboard --results "$OUT"
        --out-md "${OUT}/LEADERBOARD.md" --include-synthetic)
printf '$'
printf ' %q' "${lb_cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" -eq 0 ]]; then
    "${lb_cmd[@]}"
fi
