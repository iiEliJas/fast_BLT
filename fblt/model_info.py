"""CLI: print checkpoint metadata."""

import argparse
import json

from fblt._checkpoint import model_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    info = model_info(args.path)
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
