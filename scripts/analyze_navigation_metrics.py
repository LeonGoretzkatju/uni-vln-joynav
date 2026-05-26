#!/usr/bin/env python3
"""Compute average VLN navigation metrics from JoyNav result.json files."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_METRICS = ("success", "spl", "os", "ne", "ndtw", "steps")
AGGREGATE_ALIASES = {
    "success": "sucs_all",
    "spl": "spls_all",
    "os": "oss_all",
    "ne": "nes_all",
    "ndtw": "ndtws_all",
}


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    """Parse JSON, JSONL, or adjacent JSON objects written by append mode."""
    text = text.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    line_parse_failed = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            line_parse_failed = True
            break
        if isinstance(parsed, dict):
            records.append(parsed)

    if records and not line_parse_failed:
        return records

    decoder = json.JSONDecoder()
    records = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        parsed, end_idx = decoder.raw_decode(text, idx)
        if isinstance(parsed, dict):
            records.append(parsed)
        elif isinstance(parsed, list):
            records.extend(item for item in parsed if isinstance(item, dict))
        idx = end_idx
    return records


def numeric_value(record: dict[str, Any], metric: str) -> float | None:
    key = metric if metric in record else AGGREGATE_ALIASES.get(metric)
    if key is None or key not in record:
        return None
    value = record[key]
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def deduplicate_episode_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: OrderedDict[tuple[Any, Any, Any], dict[str, Any]] = OrderedDict()
    for record in records:
        key = (
            record.get("scene_id"),
            record.get("episode_id"),
            record.get("episode_instruction"),
        )
        deduped[key] = record
    return list(deduped.values())


def summarize(records: list[dict[str, Any]], metrics: Iterable[str], deduplicate: bool) -> dict[str, Any]:
    episode_records = [record for record in records if "success" in record or "spl" in record]
    aggregate_records = [record for record in records if any(key in record for key in AGGREGATE_ALIASES.values())]

    if episode_records:
        if deduplicate:
            episode_records = deduplicate_episode_records(episode_records)
        source_records = episode_records
        count_name = "episodes"
    else:
        source_records = aggregate_records
        count_name = "records"

    summary: dict[str, Any] = {
        count_name: len(source_records),
    }
    if episode_records:
        summary["scenes"] = len({record.get("scene_id") for record in episode_records if record.get("scene_id") is not None})

    for metric in metrics:
        values = [
            value
            for value in (numeric_value(record, metric) for record in source_records)
            if value is not None
        ]
        if values:
            summary[metric] = mean(values)

    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = ["Navigation metrics average"]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze average JoyNav VLN navigation metrics.")
    parser.add_argument(
        "json_file",
        nargs="?",
        default="/mnt/nas5/xiangchen/vlacode/JD-VLN/results/r2r/val_unseen/qwen3_sf_dyna/result.json",
        help="Path to JoyNav result.json. Defaults to qwen3_dyna_final_epoch1 result.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Metric keys to average. Default: {' '.join(DEFAULT_METRICS)}",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Keep only the latest record for each scene_id, episode_id, and instruction.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args()

    path = Path(args.json_file)
    records = parse_json_stream(path.read_text())
    summary = summarize(records, args.metrics, args.deduplicate)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=False))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
