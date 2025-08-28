#!/usr/bin/env python3
"""
Export all MongoDB collections in a database to CSV files (no CLI args).

Edit MONGO_URI, DB_NAME, and OUTPUT_DIR below, then run:
  python export_all_collections_to_csv.py
"""

import csv
import json
import os
from datetime import datetime, date
from typing import Any, Dict

from pymongo import MongoClient
from bson.objectid import ObjectId
try:
    from bson.decimal128 import Decimal128
except Exception:
    class Decimal128:  # typing fallback
        pass

# =========================
# Configuration (edit me!)
# =========================
MONGO_URI = "mongodb://localhost:28910/"
DB_NAME = "wifi_fingerprinting_data_meters"
OUTPUT_DIR = "./exports"
BATCH_SIZE = 1000
ARRAY_DELIM = "; "   # for simple lists
FIELD_SEP = "."      # for dot-notation when flattening
ENCODING = "utf-8-sig"

def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)

def _safe_json_default(o: Any) -> Any:
    if isinstance(o, (ObjectId, Decimal128)):
        return str(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)

def _to_cell(v: Any) -> str:
    """Convert common BSON/JSON types to CSV-safe string."""
    if v is None:
        return ""
    if isinstance(v, (bool, int, float, str)):
        return str(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (ObjectId, Decimal128)):
        return str(v)
    # Fallback for anything else (e.g., bytes) -> JSON/string
    try:
        return json.dumps(v, ensure_ascii=False, default=_safe_json_default)
    except Exception:
        return str(v)

def flatten_doc(d: Dict[str, Any], parent_key: str = "", sep: str = FIELD_SEP) -> Dict[str, Any]:
    """
    Recursively flatten a document.
    - Nested dicts become dot-notation keys.
    - Lists of dicts -> JSON string.
    - Lists of scalars -> joined by ARRAY_DELIM.
    """
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_doc(v, key, sep))
        elif isinstance(v, list):
            if all(isinstance(x, dict) for x in v):
                flat[key] = json.dumps(v, ensure_ascii=False, default=_safe_json_default)
            else:
                flat[key] = ARRAY_DELIM.join(_to_cell(x) for x in v)
        else:
            flat[key] = _to_cell(v)
    return flat

def export_collection(coll, out_path: str, batch_size: int = BATCH_SIZE) -> None:
    # First pass: discover headers
    headers = set()
    cursor = coll.find({}, batch_size=batch_size, no_cursor_timeout=True)
    try:
        for doc in cursor:
            headers.update(flatten_doc(doc).keys())
    finally:
        cursor.close()

    headers = sorted(headers)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Second pass: write CSV
    count = 0
    with open(out_path, "w", newline="", encoding=ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        cursor = coll.find({}, batch_size=batch_size, no_cursor_timeout=True)
        try:
            for doc in cursor:
                flat = flatten_doc(doc)
                row = {h: flat.get(h, "") for h in headers}
                writer.writerow(row)
                count += 1
        finally:
            cursor.close()

    print(f'Exported {coll.full_name} -> "{out_path}" ({count} rows, {len(headers)} columns)')

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    collections = db.list_collection_names()
    if not collections:
        print(f'No collections found in database "{DB_NAME}".')
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Found {len(collections)} collections in "{DB_NAME}". Exporting to "{OUTPUT_DIR}"...')

    for cname in collections:
        coll = db[cname]
        out_file = os.path.join(OUTPUT_DIR, f"{safe_filename(cname)}.csv")
        export_collection(coll, out_file)

    print("Done.")

if __name__ == "__main__":
    main()
