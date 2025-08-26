from pymongo import MongoClient
from bson import ObjectId
from triangle_dict import triangle_dictionary_indoor, ap_mapping
from datetime import datetime



from collections import defaultdict

def transform_wifi_data(db, triangle_name, ap_mapping, output_db,
                        dry_run=False, output_collection_name=None,
                        input_collection_name="wifi_data_indoor_global",
                        debug=False):
    """
    Transform WiFi scan data for a given triangle. For each scan:
      - Take the best (max) RSSI per BSSID, then the best per AP label.
      - If a label is missing in that scan, optionally fill it with the
        triangle-wide average RSSI for that label (computed over all scans).

    Parameters:
        db:                source Mongo DB handle
        triangle_name:     value of metadata.triangle_shape to filter
        ap_mapping:        dict {bssid_lower: "<ap_name>_rssi"}
        output_db:         destination Mongo DB handle
        dry_run:           preview without writing
        output_collection_name: destination collection name
        input_collection_name:  source collection (default: wifi_data_indoor_global)
        debug:             verbose logging
        fill_missing_with_triangle_average: if True, fill None with per-label averages
    """
    if ap_mapping is None or output_db is None or triangle_name is None:
        raise ValueError("Required parameters missing")

    if output_collection_name is None:
        raise ValueError("output_collection_name must be provided")

    collection = db[input_collection_name]
    normalized_ap_keys = set(ap_mapping.keys())
    ap_labels = set(ap_mapping.values())

    # Fetch all docs for this triangle into memory so we can do a two-pass compute
    docs = list(collection.find({"metadata.triangle_shape": triangle_name}))

    # ------------- Pass 1: compute per-label averages over the triangle -------------
    label_sum = defaultdict(float)
    label_count = defaultdict(int)

    for doc in docs:
        # Skip error docs
        if isinstance(doc.get("data"), dict) and "error" in doc["data"]:
            if debug:
                rx = doc.get("metadata", {}).get("x")
                ry = doc.get("metadata", {}).get("y")
                print(f"⚠️ Error doc at ({rx},{ry}): {doc['data'].get('error')}")
            continue

        # Best RSSI per BSSID for this scan
        bssid_to_max = {}
        for entry in doc.get("data", []):
            bssid = str(entry.get("BSSID", "")).lower()
            rssi = entry.get("RSSI")
            if bssid in normalized_ap_keys and isinstance(rssi, (int, float)):
                prev = bssid_to_max.get(bssid)
                if prev is None or rssi > prev:
                    bssid_to_max[bssid] = rssi

        # Collapse to best per AP label for this scan
        label_to_max = {}
        for bssid, rssi in bssid_to_max.items():
            label = ap_mapping[bssid]
            if (label not in label_to_max) or (rssi > label_to_max[label]):
                label_to_max[label] = rssi

        # Aggregate into sums/counts
        for label, rssi in label_to_max.items():
            label_sum[label] += float(rssi)
            label_count[label] += 1

    label_avg = {
        label: (label_sum[label] / label_count[label])
        for label in ap_labels
        if label_count[label] > 0
    }

    if debug:
        printable = {k: round(v, 2) for k, v in label_avg.items()}
        print(f"Triangle={triangle_name} averages per AP label:", printable)

    # ------------- Pass 2: build normalized results and fill missing -------------
    normalized_results = []

    for doc in docs:
        try:
            raw_x = doc["metadata"]["x"]
            raw_y = doc["metadata"]["y"]
            timestamp = doc["timestamp"]

            # Start with None for all labels
            ap_rssi = {label: None for label in ap_labels}

            # Skip error docs (keep Nones so they can be filled from averages)
            if not (isinstance(doc.get("data"), dict) and "error" in doc["data"]):
                # Best RSSI per BSSID
                bssid_to_max = {}
                for entry in doc.get("data", []):
                    bssid = str(entry.get("BSSID", "")).lower()
                    rssi = entry.get("RSSI")
                    if bssid in normalized_ap_keys and isinstance(rssi, (int, float)):
                        prev = bssid_to_max.get(bssid)
                        if prev is None or rssi > prev:
                            bssid_to_max[bssid] = rssi

                # Collapse to best per AP label
                label_to_rssis = defaultdict(list)
                for bssid, rssi in bssid_to_max.items():
                    label = ap_mapping[bssid]
                    label_to_rssis[label].append(rssi)

                for label, rssis in label_to_rssis.items():
                    if rssis:
                        ap_rssi[label] = max(rssis)

            for label in ap_labels:
                if ap_rssi[label] is None and label in label_avg:
                    ap_rssi[label] = int(round(label_avg[label]))

            new_doc = {
                "location_x": raw_x,
                "location_y": raw_y,
                "timestamp": timestamp,
                **ap_rssi
            }
            normalized_results.append(new_doc)

        except Exception as e:
            print(f"❌ Error processing document: {e}")
            continue

    # ------------- Write or preview -------------
    if dry_run:
        print(f"Dry run: Would process {len(normalized_results)} documents into {output_collection_name}")
        if normalized_results:
            print("Sample:", normalized_results[0])
        return normalized_results

    if normalized_results:
        output_db[output_collection_name].delete_many({})
        output_db[output_collection_name].insert_many(normalized_results)
        print(f"✅ Processed {len(normalized_results)} documents into {output_db.name}.{output_collection_name}")
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
        output_db = client["wifi_fingerprinting_data_raw"]
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
            debug=False
        )
