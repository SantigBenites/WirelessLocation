from pymongo import MongoClient
from datetime import datetime
from triangle_dict import triangle_dictionary, ap_mapping
from typing import Dict, Optional, Sequence
from collections import defaultdict, Counter


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
def transform_wifi_data(db, origin_x=None, origin_y=None, start_time=None, end_time=None,
                        dry_run=False, output_collection_name="wifi_data_filtered",
                        input_collection_name="wifi_data", ap_mapping=None, output_db=None, debug = False):
    """
    Transform WiFi scan data into normalized format and write to output DB/collection.
    """
    if ap_mapping is None:
        raise ValueError("ap_mapping must be provided")
    if output_db is None:
        raise ValueError("output_db must be provided")

    ip_to_y = {
        31: 1, 32: 2, 33: 3, 34: 4, 35: 5,
        36: 6, 37: 7, 38: 8, 39: 9, 30: 10
    }

    stats = Counter(total=0, error_docs=0, no_xy=0, no_ap_hits=0, bad_channel=0, emitted=0)

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
    raw_docs = list(collection.find(match_stage)) if match_stage else list(collection.find())

    # ---------- 1) Pre-compute triangle-wide averages per AP label ----------
    # We'll compute the per-scan max per BSSID, then per-label max (same logic as your per-doc path),
    # and aggregate sums/counts to get averages.
    label_sum = defaultdict(float)
    label_count = defaultdict(int)

    for doc in raw_docs:
        if isinstance(doc.get("data"), dict) and "error" in doc["data"]:
            stats["error_docs"] += 1
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
            pico_ip = doc["metadata"]["pico_ip"]
            ip_ending = int(pico_ip.split(".")[3])
            raw_y = ip_to_y.get(ip_ending)
            raw_x = doc["metadata"]["button_id"]
            timestamp = doc["timestamp"]

            if raw_y is None:
                continue

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

            new_doc = {
                "location_x": norm_x,
                "location_y": norm_y,
                "timestamp": timestamp,
                **ap_rssi,      # keep your original RSSI values
            }

            normalized_results.append(new_doc)

        except Exception as e:
            print(f"⚠️ Error processing document: {e}")
            continue

    print(stats)
    
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

    # Connect to your MongoDB
    client = MongoClient("mongodb://localhost:28910/")  # Update as needed

    # Flatten the ap_mapping: BSSID → label
    flat_ap_mapping = {
        bssid.lower(): f"{ap_name}_rssi"
        for ap_name, bssids in ap_mapping.items()
        for bssid in bssids
    }

    # Iterate through each triangle setup
    for triangle_name, current_triangle in triangle_dictionary.items():
        input_db_name       = current_triangle["db"]
        input_collection    = current_triangle["collection"]
        start_time          = current_triangle["start"]
        end_time            = current_triangle["end"]
        ap_positions        = current_triangle["ap_positions"]

        # Calculate centroid of the triangle for normalization
        origin_x, origin_y = calculate_centroid(
            *[ap_positions[ap] for ap in ["freind1", "freind2", "freind3"]]
        )

        # Select input DB
        input_db = client[input_db_name]

        # Output DB is always this
        output_db = client["wifi_fingerprinting_data_exponential"]

        # Collection name will match triangle name (e.g., reto_grande_wifi_client_data_global)
        output_collection = triangle_name

        # Run transform
        transform_wifi_data(
            db=input_db,
            origin_x=origin_x,
            origin_y=origin_y,
            start_time=start_time,
            end_time=end_time,
            dry_run=True,
            input_collection_name=input_collection,
            output_collection_name=output_collection,
            ap_mapping=flat_ap_mapping,
            output_db=output_db,
            debug=True
        )