from pymongo import MongoClient
from datetime import datetime
from triangle_dict import triangle_dictionary, ap_mapping


import sys
from collections import defaultdict

def transform_wifi_data(db, start_time=None, end_time=None,
                        dry_run=False, output_collection_name="wifi_data_filtered",
                        input_collection_name="wifi_data", ap_mapping=None, output_db=None,
                        debug=False):
    """
    Transform WiFi scan data into normalized format and write to output DB/collection.
    Also fills missing AP RSSI values with the triangle-wide average for that AP.
    """
    if ap_mapping is None:
        raise ValueError("ap_mapping must be provided")
    if output_db is None:
        raise ValueError("output_db must be provided")

    ip_to_y = {
        31: 1, 32: 2, 33: 3, 34: 4, 35: 5,
        36: 6, 37: 7, 38: 8, 39: 9, 30: 10
    }

    # Build time match
    match_stage = {}
    if start_time:
        match_stage["timestamp"] = {"$gte": start_time.timestamp()}
    if end_time:
        match_stage.setdefault("timestamp", {})["$lte"] = end_time.timestamp()

    collection = db[input_collection_name]
    normalized_ap_keys = set(ap_mapping.keys())     # set of BSSIDs we care about
    ap_labels = set(ap_mapping.values())            # e.g., {"freind1_rssi", "freind2_rssi", ...}

    # Pull docs for this triangle window
    raw_docs = list(collection.find(match_stage)) if match_stage else list(collection.find())

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
            pico_ip = doc["metadata"]["pico_ip"]
            ip_ending = int(pico_ip.split(".")[3])
            raw_y = ip_to_y.get(ip_ending)
            raw_x = doc["metadata"]["button_id"]
            timestamp = doc["timestamp"]

            if raw_y is None:
                # Unknown Pico -> skip
                continue

            # Start with None for all labels
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

            new_doc = {
                "location_x": raw_x,
                "location_y": raw_y,
                "timestamp": timestamp,
                **ap_rssi
            }

            normalized_results.append(new_doc)

        except Exception as e:
            print(f"⚠️ Error processing document: {e}")
            continue

    # ---------- 3) Write or preview ----------
    if dry_run:
        print(f"Dry run: Would process {len(normalized_results)} documents")
        print(f"Documents would be written to {output_db.name}.{output_collection_name}")
        if normalized_results:
            doc = normalized_results[0]
            print(f"  location: {doc['location_x']:.4f},{doc['location_y']:.4f}")
            from datetime import datetime as _dt
            print(f"  timestamp: {_dt.fromtimestamp(doc['timestamp'])}")
            shown = set()
            for label in ap_labels:
                if label not in shown:
                    print(f"  {label}: {doc.get(label, 'N/A')}")
                    shown.add(label)
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


        # Select input DB
        input_db = client[input_db_name]

        # Output DB is always this
        output_db = client["wifi_fingerprinting_data_raw"]

        # Collection name will match triangle name (e.g., reto_grande_wifi_client_data_global)
        output_collection = triangle_name

        # Run transform
        transform_wifi_data(
            db=input_db,
            start_time=start_time,
            end_time=end_time,
            dry_run=False,
            input_collection_name=input_collection,
            output_collection_name=output_collection,
            ap_mapping=flat_ap_mapping,
            output_db=output_db,
            debug=True
        )