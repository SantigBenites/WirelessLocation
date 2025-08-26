from pymongo import MongoClient
from datetime import datetime
from triangle_dict import triangle_dictionary_indoor, ap_mapping
from typing import Dict, Optional, Sequence
from collections import defaultdict


def _pairwise_ratios(values: Dict[str, Optional[float]], cols: Sequence[str]) -> Dict[str, Optional[float]]:
    """
    Build {f"{a}_over_{b}": values[a]/values[b]} for all a != b in cols.
    Returns None when denominator is 0 or either value is None.
    """
    out: Dict[str, Optional[float]] = {}
    for a in cols:
        for b in cols:
            if a == b:
                continue
            va = values.get(a)
            vb = values.get(b)
            key = f"{a}_over_{b}"
            out[key] = (va / vb) if (va is not None and vb not in (None, 0)) else None
    return out

def normalize_picos_coordinates(x, y, origin_x, origin_y):

 
    # Normlized interval sizes
    pico_interval = 1 / 10 
    ap_interval = 1/4   

    # Normalized locations
    normalized_x = pico_interval * x
    normalized_y = pico_interval * y
    normalized_origin_x = ap_interval * origin_x
    normalized_origin_y = ap_interval * origin_y
    
    
    return (normalized_x-normalized_origin_x, normalized_y-normalized_origin_y)

def calculate_centroid(point1, point2, point3):
    cx = (point1[0] + point2[0] + point3[0]) / 3
    cy = (point1[1] + point2[1] + point3[1]) / 3
    return (cx, cy)

import sys
def transform_wifi_data(db, triangle_name, origin_x=None, origin_y=None, start_time=None, end_time=None,
                        dry_run=False, output_collection_name="wifi_data_filtered",
                        input_collection_name="wifi_data", ap_mapping=None, output_db=None, debug = False):
    """
    Transform WiFi scan data into normalized format and write to output DB/collection.
    """
    if ap_mapping is None or output_db is None or triangle_name is None:
        raise ValueError("Required parameters missing")

    if output_collection_name is None:
        raise ValueError("output_collection_name must be provided")

    ip_to_y = {
        31: 1, 32: 2, 33: 3, 34: 4, 35: 5,
        36: 6, 37: 7, 38: 8, 39: 9, 30: 10
    }

    match_stage = {}
    if start_time:
        match_stage["timestamp"] = {"$gte": start_time.timestamp()}
    if end_time:
        match_stage.setdefault("timestamp", {})["$lte"] = end_time.timestamp()

    collection = db[input_collection_name]
    normalized_ap_keys = set(ap_mapping.keys())
    ap_labels = set(ap_mapping.values())

    if match_stage:
        raw_docs = list(collection.find(match_stage))
    else:
        raw_docs = list(collection.find())

    # Pull docs for this triangle window
    raw_docs = list(collection.find({"metadata.triangle_shape": triangle_name}))

    # ---------- 1) Pre-compute triangle-wide averages per AP label ----------
    # We'll compute the per-scan max per BSSID, then per-label max (same logic as your per-doc path),
    # and aggregate sums/counts to get averages.
    label_sum = defaultdict(float)
    label_count = defaultdict(int)

    for doc in raw_docs:
        if isinstance(doc.get("data"), dict) and "error" in doc["data"]:
            continue

        # Map BSSID -> max RSSI in this scan
        bssid_to_max_rssi = {}
        for entry in doc.get("data", []):
            bssid = str(entry.get("BSSID", "")).lower()
            rssi = entry.get("RSSI")
            if bssid in normalized_ap_keys and isinstance(rssi, (int, float)):
                prev = bssid_to_max_rssi.get(bssid)
                if prev is None or rssi > prev:
                    bssid_to_max_rssi[bssid] = rssi

        # Collapse BSSID values into label values via "max per AP label"
        label_to_rssi = {}
        for bssid, label in ap_mapping.items():
            rssi = bssid_to_max_rssi.get(bssid)
            if rssi is not None:
                if label not in label_to_rssi or rssi > label_to_rssi[label]:
                    label_to_rssi[label] = rssi

        # Aggregate into sums/counts
        for label, rssi in label_to_rssi.items():
            label_sum[label] += float(rssi)
            label_count[label] += 1

    # Final averages per label (only where we have data)
    label_avg = {
        label: (label_sum[label] / label_count[label])
        for label in ap_labels
        if label_count[label] > 0
    }

    if debug:
        print("Triangle averages per AP label:", {k: round(v, 2) for k, v in label_avg.items()})

    # ---------- 2) Build normalized results and fill missing with averages ----------
    normalized_results = []

    for doc in raw_docs:
        try:
            raw_x = doc["metadata"]["x"]
            raw_y = doc["metadata"]["y"]
            timestamp = doc["timestamp"]

            norm_x, norm_y = normalize_picos_coordinates(
                raw_x, raw_y,
                origin_x if origin_x is not None else 0,
                origin_y if origin_y is not None else 0
            )

            ap_rssi = {label: None for label in ap_labels}

            # Extract this scan's best RSSI per label (same logic as above)
            if not (isinstance(doc.get("data"), dict) and "error" in doc["data"]):
                bssid_to_max_rssi = {}
                for entry in doc.get("data", []):
                    bssid = str(entry.get("BSSID", "")).lower()
                    rssi = entry.get("RSSI")
                    if bssid in normalized_ap_keys and isinstance(rssi, (int, float)):
                        prev = bssid_to_max_rssi.get(bssid)
                        if prev is None or rssi > prev:
                            bssid_to_max_rssi[bssid] = rssi

                # Collapse into label max
                label_to_rssis = defaultdict(list)
                for bssid, label in ap_mapping.items():
                    rssi = bssid_to_max_rssi.get(bssid)
                    if rssi is not None:
                        label_to_rssis[label].append(rssi)

                for label, rssis in label_to_rssis.items():
                    if rssis:
                        ap_rssi[label] = max(rssis)

            for label in ap_labels:
                if ap_rssi[label] is None and label in label_avg:
                    # Keep integer-like RSSI; round the average
                    ap_rssi[label] = int(round(label_avg[label]))

            # 👉 NEW: compute pairwise RSSI ratios for the three friends
            rssi_cols_fixed = ["freind1_rssi", "freind2_rssi", "freind3_rssi"]
            ratio_cols = _pairwise_ratios(ap_rssi, rssi_cols_fixed)

            new_doc = {
                "location_x": norm_x,
                "location_y": norm_y,
                "timestamp": timestamp,
                **ap_rssi,      # keep your original RSSI values
                **ratio_cols    # add ratio features like freind1_rssi_over_freind2_rssi, etc.
            }

            normalized_results.append(new_doc)

        except Exception as e:
            print(f"⚠️ Error processing document: {e}")
            continue

    if dry_run:
        print(f"Dry run: Would process {len(normalized_results)} documents")
        print(f"Documentos would be processed into {output_db.name}.{output_collection_name}")
        if normalized_results:
            doc = normalized_results[0]
            print(f"  Normalized location: {doc['location_x']:.4f},{doc['location_y']:.4f}")
            print(f"  timestamp: {datetime.fromtimestamp(doc['timestamp'])}")
            seen = set()
            for label in ap_labels:
                if label not in seen:
                    print(f"  {label}: {doc.get(label, 'N/A')}")
                    seen.add(label)
        return normalized_results

    if normalized_results:
        output_db[output_collection_name].delete_many({})
        output_db[output_collection_name].insert_many(normalized_results)
        print(f"✅ Processed {len(normalized_results)} documents into "
              f"{output_db.name}.{output_collection_name}")
        return normalized_results
    else:
        print("No documents matched the criteria")
        return []




if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:28910/")

    flat_ap_mapping = {
        bssid.lower(): f"{ap_name}_rssi"
        for ap_name, bssids in ap_mapping.items()
        for bssid in bssids
    }

    for key, triangle in triangle_dictionary_indoor.items():
        triangle_name = triangle["triangle_name"]
        db = client[triangle["db"]]
        output_db = client["wifi_fingerprinting_data_exponential"]
        input_collection = triangle["collection"]
        output_collection = key

        ap_positions = triangle["ap_positions"]

        transform_wifi_data(
            db=db,
            triangle_name=triangle_name,
            ap_mapping=flat_ap_mapping,
            output_db=output_db,
            input_collection_name=input_collection,
            output_collection_name=output_collection,
            dry_run=False,
            debug=True
        )