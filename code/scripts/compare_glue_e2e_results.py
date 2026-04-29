import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.generation_metrics import PAPER_E2E_LORA
from scripts.compare_roberta_results import (
    DEFAULT_AGGREGATE,
    DEFAULT_RESULTS_ROOT,
    PAPER_RESULTS,
    PRIMARY_METRIC,
    TASK_ORDER,
    extract_best_metric,
    load_json,
    load_task_result,
)


DEFAULT_E2E_RESULTS_CANDIDATES = [
    REPO_ROOT / "results" / "GPT2-M" / "e2e_eval" / "results.json",
    REPO_ROOT / "results" / "e2e_eval" / "results.json",
]

E2E_METRIC_ORDER = ["BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr"]
E2E_PERCENT_METRICS = {"BLEU", "METEOR", "ROUGE_L"}


def resolve_e2e_results_path(explicit_path: Path | None):
    if explicit_path is not None:
        return explicit_path
    for candidate in DEFAULT_E2E_RESULTS_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_E2E_RESULTS_CANDIDATES[0]


def format_float(value, digits=1):
    return f"{value:.{digits}f}" if value is not None else "–"


def format_signed(value, digits=1):
    return f"{value:+.{digits}f}" if value is not None else "–"


def render_table(rows, headers):
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    body = ["  ".join(row[header].ljust(widths[header]) for header in headers) for row in rows]
    return "\n".join([header_line, "  ".join("-" * widths[header] for header in headers), *body])


def scale_e2e_metric(metric_name, value):
    if value is None:
        return None
    return value * 100.0 if metric_name in E2E_PERCENT_METRICS else value


def build_glue_rows(results_root: Path, aggregate: dict | None):
    rows = []
    csv_rows = []

    for task in TASK_ORDER:
        result = load_task_result(task, results_root, aggregate)
        best_metric, metric_name = extract_best_metric(task, result)
        paper = PAPER_RESULTS[task]
        ours = best_metric * 100 if best_metric is not None else None
        delta = ours - paper if ours is not None else None
        wall_seconds = result.get("wall_seconds") if result else None
        wall_minutes = wall_seconds / 60.0 if wall_seconds is not None else None

        row = {
            "task": task.upper(),
            "metric": metric_name,
            "paper": format_float(paper, digits=1),
            "ours": format_float(ours, digits=1),
            "delta": format_signed(delta, digits=1),
            "wall_min": format_float(wall_minutes, digits=1),
        }
        rows.append(row)
        csv_rows.append({"benchmark": "GLUE", **row})

    return rows, csv_rows


def build_e2e_rows(e2e_results_path: Path):
    if not e2e_results_path.exists():
        rows = []
        csv_rows = []
        for metric_name in E2E_METRIC_ORDER:
            row = {
                "task": "E2E",
                "metric": metric_name,
                "paper": format_float(scale_e2e_metric(metric_name, PAPER_E2E_LORA[metric_name]), digits=2),
                "ours": "–",
                "delta": "–",
                "backend": "–",
            }
            rows.append(row)
            csv_rows.append({"benchmark": "E2E", **row})
        return rows, csv_rows

    result = load_json(e2e_results_path)
    ours_metrics = result.get("metrics", {})
    paper_metrics = result.get("paper", PAPER_E2E_LORA)
    backend = result.get("scoring_backend", "unknown")

    rows = []
    csv_rows = []
    for metric_name in E2E_METRIC_ORDER:
        paper = scale_e2e_metric(metric_name, paper_metrics.get(metric_name))
        ours = scale_e2e_metric(metric_name, ours_metrics.get(metric_name))
        delta = ours - paper if ours is not None and paper is not None else None
        digits = 2
        row = {
            "task": "E2E",
            "metric": metric_name,
            "paper": format_float(paper, digits=digits),
            "ours": format_float(ours, digits=digits),
            "delta": format_signed(delta, digits=digits),
            "backend": backend,
        }
        rows.append(row)
        csv_rows.append({"benchmark": "E2E", **row})

    return rows, csv_rows


def main():
    parser = argparse.ArgumentParser(
        description="Display GLUE and E2E results against the LoRA paper's reported numbers."
    )
    parser.add_argument("--roberta-results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--roberta-aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--e2e-results", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    aggregate = load_json(args.roberta_aggregate) if args.roberta_aggregate.exists() else None
    e2e_results_path = resolve_e2e_results_path(args.e2e_results)

    glue_rows, glue_csv_rows = build_glue_rows(args.roberta_results_root, aggregate)
    e2e_rows, e2e_csv_rows = build_e2e_rows(e2e_results_path)

    print("=== GLUE (RoBERTa-base, paper Table 2) ===")
    print(render_table(glue_rows, headers=["task", "metric", "paper", "ours", "delta", "wall_min"]))
    print("\n=== E2E NLG (GPT-2 Medium, paper Table 3) ===")
    print(render_table(e2e_rows, headers=["task", "metric", "paper", "ours", "delta", "backend"]))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["benchmark", "task", "metric", "paper", "ours", "delta", "wall_min", "backend"],
            )
            writer.writeheader()
            for row in glue_csv_rows:
                writer.writerow({**row, "backend": "", "wall_min": row["wall_min"]})
            for row in e2e_csv_rows:
                writer.writerow({**row, "wall_min": ""})
        print(f"\nWrote CSV to {args.csv_out}")


if __name__ == "__main__":
    main()
