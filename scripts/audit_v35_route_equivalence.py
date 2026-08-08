#!/usr/bin/env python
"""Audit whether Task 2 clicks preserve the Task 1 v3.5 validation routes.

When click injection is disabled, clicks affect Task 2 only through the forced
anatomy tuple. If every click strategy selects the same tuple used to generate
the Task 1 v3.5 predictions, the Task 1 proxy is a configuration-equivalent
estimate for the Task 2 candidate without another full-cohort GPU pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "inference" / "inference.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report-root", type=Path, required=True)
    parser.add_argument("--click-root", type=Path, required=True)
    parser.add_argument("--task1-evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("task2_entrypoint", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    entrypoint = load_entrypoint()
    reports = sorted(args.gate_report_root.glob("case_*.json"))
    strategies = sorted(path.name for path in args.click_root.iterdir() if path.is_dir())
    if not reports:
        raise RuntimeError(f"no gate reports found under {args.gate_report_root}")
    if not strategies:
        raise RuntimeError(f"no click strategies found under {args.click_root}")

    rows = []
    mismatches = []
    for report_path in reports:
        report = json.loads(report_path.read_text())
        case_id = str(report["case_id"])
        expected = tuple(report["anatomies"])
        for strategy in strategies:
            click_path = (
                args.click_root
                / strategy
                / case_id
                / "peripelvic-fragment-clicks.json"
            )
            points = entrypoint.load_clicks(click_path)
            routing = entrypoint.route_from_clicks(points)
            forced = entrypoint.anatomies_from_routing(routing)
            row = {
                "case_id": case_id,
                "strategy": strategy,
                "click_count": len(points),
                "family": routing["family"],
                "expected_anatomies": list(expected),
                "forced_anatomies": list(forced) if forced is not None else None,
                "matches": forced == expected,
            }
            rows.append(row)
            if not row["matches"]:
                mismatches.append(row)

    task1_evaluation = json.loads(args.task1_evaluation.read_text())
    equivalent = not mismatches
    report = {
        "protocol": "task2_v35_click_route_equivalence_proxy_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "task2-v3.5-always-expert-t075",
        "configuration": {
            "task1_base": "v3.5-always-expert-t075",
            "click_injection": False,
            "stage_b_policy": "always anatomy expert",
            "agglomeration_threshold": 0.75,
        },
        "audit": {
            "n_cases": len(reports),
            "n_strategies": len(strategies),
            "n_case_strategy_pairs": len(rows),
            "matching_pairs": len(rows) - len(mismatches),
            "mismatch_count": len(mismatches),
            "all_routes_equivalent": equivalent,
            "strategies": strategies,
        },
        "transferred_evaluation": {
            "valid": equivalent,
            "derivation": (
                "Task 2 clicks only force the anatomy tuple; click splitting is off. "
                "All audited tuples match those used for the deterministic Task 1 "
                "v3.5 experiment, so its metrics are transferred as a route-equivalent "
                "proxy estimate rather than a new Task 2 inference run."
            ),
            "source_protocol": task1_evaluation.get("protocol"),
            "n_cases": task1_evaluation.get("n_cases"),
            "n_anatomy_samples": task1_evaluation.get("n_anatomy_samples"),
            "overall": task1_evaluation.get("overall") if equivalent else None,
            "by_anatomy": task1_evaluation.get("by_anatomy") if equivalent else None,
        },
        "limitations": [
            "This is a route-equivalent estimate from the existing fold-0 proxy, not a full-cohort Task 2 rerun or a Grand Challenge hidden-test score.",
            "The official PENGWIN evaluator is unavailable; metrics use the documented official-aligned proxy-v2 implementation.",
            "This route audit does not replace the separate GPU container smoke test recorded in the release.",
        ],
        "mismatches": mismatches,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"audit": report["audit"], "overall": report["transferred_evaluation"]["overall"]}, indent=2))
    if not equivalent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
