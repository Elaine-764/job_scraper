#!/usr/bin/env python3
"""Combine every file in roles/ into a single CSV.

Usage:
    python combine_roles.py [roles_dir] [output_csv]

Defaults:
    roles_dir -> ./roles
    output_csv -> ./combined_roles.csv

This script handles JSON (list or single object), NDJSON/JSONL, and CSV files.
Non-scalar JSON values are JSON-encoded into CSV cells.
"""
from pathlib import Path
import sys
import json
import csv
from typing import Iterator


def normalize_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, ensure_ascii=False)


def iter_records_from_file(path: Path) -> Iterator[dict]:
    ext = path.suffix.lower()
    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Common pattern: {"company": "X", "listings": [...]}
                    if "listings" in item and isinstance(item.get("listings"), list):
                        company = item.get("company")
                        for listing in item.get("listings", []):
                            if isinstance(listing, dict):
                                out = dict(listing)
                                if company is not None:
                                    out.setdefault("company", company)
                                yield out
                            else:
                                yield {"value": listing}
                    else:
                        yield item
                else:
                    yield {"value": item}
        elif isinstance(data, dict):
            # If a dict contains a single top-level list-of-dicts, yield those
            for v in data.values():
                if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                    for item in v:
                        yield item
                    return
            # Otherwise treat the dict as one record
            yield data
        else:
            yield {"value": data}

    elif ext in (".ndjson", ".jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    yield {"line": line}
                else:
                    yield obj if isinstance(obj, dict) else {"value": obj}

    elif ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    else:
        return


def main():
    roles_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("roles")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("combined_roles.csv")

    if not roles_dir.exists() or not roles_dir.is_dir():
        print(f"Roles directory not found: {roles_dir}")
        sys.exit(1)

    rows = []
    # enforce the exact columns requested
    fieldnames = ["title", "location", "deadline", "url", "team", "company"]

    for p in sorted(roles_dir.iterdir()):
        if not p.is_file():
            continue
        for rec in iter_records_from_file(p):
            if not isinstance(rec, dict):
                rec = {"value": rec}
            # Only keep the requested columns, coerce values
            flat = {k: normalize_value(rec.get(k)) if k in rec or k == "company" else "" for k in fieldnames}
            # If company missing, try to set from outer file metadata: if top-level rec had company it's already set; else try filename
            if not flat.get("company"):
                # derive company from filename (strip suffix)
                derived = p.stem
                flat["company"] = derived
            rows.append(flat)

    if not rows:
        print("No records found in roles/ to combine.")
        sys.exit(0)

    # Write CSV using the exact requested field order
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
