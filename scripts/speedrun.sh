#!/usr/bin/env bash
# Repo-local launcher for `python3 -m fblt.speedrun`.
#
# --help, --dry-run, --backend, --out, --steps and --build/--no-build are
# recognised here; every other argument is forwarded verbatim. Default OUT is
# `runs/speedrun` (override with the OUT environment variable). Build stays ON
# unless the caller passes --no-build, which is what the Python CLI defaults to.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/speedrun.sh [--out DIR] [--backend cpu|cuda] [--steps N]
                           [--dry-run] [--build|--no-build] [-- ARGS...]

Runs one reproducible speedrun from the repository root. All unrecognised
arguments are passed through to `python3 -m fblt.speedrun`.

  -h, --help      Show this help and exit.
  --dry-run       Print the planned commands without creating anything.
  --backend B     cpu (default) or cuda.
  --out DIR       Output directory (default: runs/speedrun).
  --steps N       Training steps.
  --build         Configure and compile first (default).
  --no-build      Reuse binaries that are already built.

Environment: OUT overrides the default output directory.
EOF
}

OUT="${OUT:-runs/speedrun}"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "speedrun.sh: --out needs a value" >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        --out=*)
            OUT="${1#*=}"
            shift
            ;;
        --backend|--steps)
            [[ $# -ge 2 ]] || { echo "speedrun.sh: $1 needs a value" >&2; exit 2; }
            PASSTHRU+=("$1" "$2")
            shift 2
            ;;
        --dry-run|--build|--no-build)
            PASSTHRU+=("$1")
            shift
            ;;
        *)
            PASSTHRU+=("$1")
            shift
            ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec python3 -m fblt.speedrun --out "$OUT" "${PASSTHRU[@]}"
