#!/usr/bin/env python3
"""Generate CoAX cognitive-parameter ranges from the fitted parameter CSV.

This script intentionally uses only the Python standard library so it can run
even when the local scientific Python stack is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "assets" / "param_config" / "CoAX_cog_param.csv"
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "demo" / "coax_cognitive_parameter_ranges.json"

COAX_COGNITIVE_PARAMETER_COLUMNS = [
    "k",
    "sensitivity",
    "retrieval_threshold",
    "scaling_factor",
    "decay_param",
]

INTEGER_PARAMETERS = {"k"}


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def nice_step(minimum: float, maximum: float, is_integer: bool) -> float:
    if is_integer:
        return 1

    span = abs(maximum - minimum)
    if span == 0:
        return max(abs(minimum) / 100, 0.001)

    raw_step = span / 100
    magnitude = 10 ** math.floor(math.log10(raw_step))
    return max(round(raw_step / magnitude) * magnitude, magnitude)


def infer_range(values: Sequence[float], name: str) -> Dict[str, Any]:
    minimum = min(values)
    maximum = max(values)
    is_integer = name in INTEGER_PARAMETERS or all(float(value).is_integer() for value in values)
    default = median(values)

    if is_integer:
        minimum = int(math.floor(minimum))
        maximum = int(math.ceil(maximum))
        default = int(round(default))

    return {
        "min": minimum,
        "max": maximum,
        "default": default,
        "step": nice_step(float(minimum), float(maximum), is_integer),
        "type": "integer" if is_integer else "float",
        "count": len(values),
    }


def collect_values(
    rows: Sequence[Dict[str, str]],
    columns: Iterable[str],
    app_id: Optional[str] = None,
    xai_type: Optional[str] = None,
) -> Dict[str, List[float]]:
    values_by_column: Dict[str, List[float]] = {column: [] for column in columns}
    for row in rows:
        if app_id is not None and row.get("appId") != app_id:
            continue
        if xai_type is not None and row.get("XAIType") != xai_type:
            continue
        for column in columns:
            value = finite_float(row.get(column))
            if value is not None:
                values_by_column[column].append(value)
    return values_by_column


def build_range_map(values_by_column: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    return {
        column: infer_range(values, column)
        for column, values in values_by_column.items()
        if values
    }


def generate_ranges(input_csv: Path, output_json: Path) -> Dict[str, Any]:
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    parameter_columns = [
        column
        for column in COAX_COGNITIVE_PARAMETER_COLUMNS
        if column in fieldnames
    ]
    app_ids = sorted(
        {
            row.get("appId", "").strip()
            for row in rows
            if row.get("appId", "").strip()
        }
    )
    xai_types = sorted(
        {
            row.get("XAIType", "").strip()
            for row in rows
            if row.get("XAIType", "").strip()
        }
    )

    global_ranges = build_range_map(collect_values(rows, parameter_columns))
    ranges_by_app = {
        app_id: build_range_map(collect_values(rows, parameter_columns, app_id=app_id))
        for app_id in app_ids
    }
    ranges_by_app_xai = {
        app_id: {
            xai_type: build_range_map(
                collect_values(rows, parameter_columns, app_id=app_id, xai_type=xai_type)
            )
            for xai_type in xai_types
            if any(row.get("appId") == app_id and row.get("XAIType") == xai_type for row in rows)
        }
        for app_id in app_ids
    }

    payload = {
        "schema_version": 1,
        "source": {
            "csv": str(input_csv.relative_to(REPO_ROOT)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(rows),
        },
        "parameter_columns": list(global_ranges),
        "app_ids": app_ids,
        "xai_types": xai_types,
        "global": global_ranges,
        "by_app": ranges_by_app,
        "by_app_xai": ranges_by_app_xai,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = generate_ranges(args.input, args.output)

    print(f"Wrote {args.output}")
    print(f"Source rows: {payload['source']['row_count']}")
    print(f"App IDs: {', '.join(payload['app_ids'])}")
    print(f"XAI types: {', '.join(payload['xai_types'])}")
    print(f"Parameters: {', '.join(payload['parameter_columns'])}")


if __name__ == "__main__":
    main()
