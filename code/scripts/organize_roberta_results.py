import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE = REPO_ROOT / "results" / "myLoRA_results.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "RoBERTa"


def main():
    parser = argparse.ArgumentParser(
        description="Split results/myLoRA_results.json into results/RoBERTa/<task>/result.json files."
    )
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with args.aggregate.open() as f:
        aggregate = json.load(f)

    if not isinstance(aggregate, dict):
        raise ValueError(f"Expected top-level dict in {args.aggregate}")

    written = 0
    skipped = 0

    for task in sorted(aggregate):
        result = aggregate[task]
        out_dir = args.output_root / task
        out_path = out_dir / "result.json"

        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} already exists")
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"[write] {out_path}")
        written += 1

    print(f"done: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
