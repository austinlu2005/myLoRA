import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "RoBERTa"
DEFAULT_AGGREGATE = REPO_ROOT / "results" / "myLoRA_results.json"

TASK_ORDER = ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]

PAPER_RESULTS = {
    "mnli": 87.5,
    "sst2": 95.1,
    "mrpc": 89.7,
    "cola": 63.4,
    "qnli": 93.3,
    "qqp": 90.8,
    "rte": 86.6,
    "stsb": 91.5,
}

PRIMARY_METRIC = {
    "mnli": "accuracy",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "cola": "matthews_correlation",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "stsb": "pearson",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_task_result(task: str, results_root: Path, aggregate: dict | None):
    path = results_root / task / "result.json"
    if path.exists():
        return load_json(path)
    if aggregate and task in aggregate:
        return aggregate[task]
    return None


def extract_best_metric(task: str, result: dict | None):
    if result is None:
        return None, PRIMARY_METRIC[task]

    metric_name = result.get("metric_name") or PRIMARY_METRIC[task]
    best_metric = result.get("best_metric")

    if best_metric is None:
        history = result.get("history", [])
        values = [row[metric_name] for row in history if metric_name in row]
        best_metric = max(values) if values else None

    return best_metric, metric_name


def render_table(rows):
    headers = ["task", "metric", "paper", "ours", "delta", "wall_min"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "  ".join("-" * widths[header] for header in headers)
    body = ["  ".join(row[header].ljust(widths[header]) for header in headers) for row in rows]
    return "\n".join([header_line, sep_line, *body])


def main():
    parser = argparse.ArgumentParser(
        description="Display RoBERTa GLUE results against the LoRA paper's Table 2 row."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    aggregate = load_json(args.aggregate) if args.aggregate.exists() else None

    rows = []
    csv_rows = []
    for task in TASK_ORDER:
        result = load_task_result(task, args.results_root, aggregate)
        best_metric, metric_name = extract_best_metric(task, result)
        paper = PAPER_RESULTS[task]
        ours = best_metric * 100 if best_metric is not None else None
        delta = ours - paper if ours is not None else None
        wall_seconds = result.get("wall_seconds") if result else None
        wall_minutes = wall_seconds / 60.0 if wall_seconds is not None else None

        row = {
            "task": task.upper(),
            "metric": metric_name,
            "paper": f"{paper:.1f}",
            "ours": f"{ours:.1f}" if ours is not None else "–",
            "delta": f"{delta:+.1f}" if delta is not None else "–",
            "wall_min": f"{wall_minutes:.1f}" if wall_minutes is not None else "–",
        }
        rows.append(row)
        csv_rows.append(
            {
                "task": task,
                "metric": metric_name,
                "paper": f"{paper:.1f}",
                "ours": f"{ours:.1f}" if ours is not None else "",
                "delta": f"{delta:+.1f}" if delta is not None else "",
                "wall_min": f"{wall_minutes:.1f}" if wall_minutes is not None else "",
            }
        )

    print(render_table(rows))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["task", "metric", "paper", "ours", "delta", "wall_min"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote CSV to {args.csv_out}")


if __name__ == "__main__":
    main()
