from pymongo import MongoClient
from bson import ObjectId
from triangle_dict import triangle_dictionary_indoor, ap_mapping
from datetime import datetime


def normalize_picos_coordinates(x, y, origin_x, origin_y):
    pico_interval = 1 / 5
    ap_interval = 1 / 5

    normalized_x = pico_interval * x
    normalized_y = pico_interval * y
    normalized_origin_x = ap_interval * origin_x
    normalized_origin_y = ap_interval * origin_y

    return (normalized_x - normalized_origin_x, normalized_y - normalized_origin_y)


def calculate_centroid(point1, point2, point3):
    cx = (point1[0] + point2[0] + point3[0]) / 3
    cy = (point1[1] + point2[1] + point3[1]) / 3
    return (cx, cy)


def transform_wifi_data(db, triangle_name, origin_x, origin_y, ap_mapping, output_db,
                        dry_run=False, output_collection_name=None,
                        input_collection_name="wifi_data_indoor_global", debug=False):
    if ap_mapping is None or output_db is None or triangle_name is None:
        raise ValueError("Required parameters missing")

    collection = db[input_collection_name]
    normalized_ap_keys = list(ap_mapping.keys())
    ap_labels = set(ap_mapping.values())

    cursor = collection.find({"metadata.triangle_shape": triangle_name})

    from collections import defaultdict

    normalized_results = []

    for doc in cursor:
        try:
            raw_x = doc["metadata"]["x"]
            raw_y = doc["metadata"]["y"]
            timestamp = doc["timestamp"]

            norm_x, norm_y = normalize_picos_coordinates(raw_x, raw_y, origin_x, origin_y)

            ap_rssi = {label: None for label in ap_labels}  # default to None

            if isinstance(doc.get("data"), dict) and "error" in doc["data"]:
                # Log error if needed
                if debug:
                    print(f"⚠️ Error in document at ({raw_x},{raw_y}): {doc['data']['error']}")
            else:
                bssid_to_rssi = {}
                for entry in doc.get("data", []):
                    bssid = entry.get("BSSID", "").lower()
                    rssi = entry.get("RSSI")
                    if bssid in normalized_ap_keys:
                        if bssid not in bssid_to_rssi or rssi > bssid_to_rssi[bssid]:
                            bssid_to_rssi[bssid] = rssi

                label_to_rssis = defaultdict(list)
                for bssid, rssi in bssid_to_rssi.items():
                    label = ap_mapping[bssid]
                    label_to_rssis[label].append(rssi)

                for label in ap_labels:
                    if label_to_rssis[label]:
                        ap_rssi[label] = max(label_to_rssis[label])

            new_doc = {
                "location_x": norm_x,
                "location_y": norm_y,
                "timestamp": timestamp,
                **ap_rssi
            }

            normalized_results.append(new_doc)

        except Exception as e:
            print(f"❌ Error processing document: {e}")
            continue

    if dry_run:
        print(f"Dry run: Would process {len(normalized_results)} documents")
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
        output_db = client["wifi_fingerprinting_data"]
        input_collection = triangle["collection"]
        output_collection = key

        ap_positions = triangle["ap_positions"]
        origin_x, origin_y = calculate_centroid(
            ap_positions["freind1"], ap_positions["freind2"], ap_positions["freind3"]
        )

        transform_wifi_data(
            db=db,
            triangle_name=triangle_name,
            origin_x=origin_x,
            origin_y=origin_y,
            ap_mapping=flat_ap_mapping,
            output_db=output_db,
            input_collection_name=input_collection,
            output_collection_name=output_collection,
            dry_run=False
        )
