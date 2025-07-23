from pymongo import MongoClient
from datetime import datetime
from triangle_dict import triangle_dictionary, ap_mapping


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


def transform_wifi_data(db, origin_x=None, origin_y=None, start_time=None, end_time=None,
                        dry_run=False, output_collection_name="wifi_data_filtered",
                        input_collection_name="wifi_data", ap_mapping=None, output_db=None):
    """
    Transform WiFi scan data into normalized format and write to output DB/collection.
    
    Args:
        db: MongoDB *input* database object
        output_db: MongoDB *output* database object
        ...
    """
    if ap_mapping is None:
        raise ValueError("ap_mapping must be provided")
    if output_db is None:
        raise ValueError("output_db must be provided")

    
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

    pipeline = [
        {"$match": match_stage} if match_stage else {"$match": {}},
        {
            "$addFields": {
                "ip_ending": {
                    "$toInt": {"$arrayElemAt": [{"$split": ["$metadata.pico_ip", "."]}, 3]}
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "raw_location_x": "$metadata.button_id",
                "raw_location_y": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$ip_ending", val]}, "then": ip_to_y[val]}
                            for val in ip_to_y
                        ],
                        "default": None
                    }
                },
                "data": 1,
                "timestamp": 1
            }
        },
        {"$match": {"raw_location_y": {"$ne": None}}},
        {"$unwind": "$data"},
        {"$match": {"data.BSSID": {"$in": list(ap_mapping.keys())}}},
        {
            "$group": {
                "_id": {
                    "raw_location_x": "$raw_location_x",
                    "raw_location_y": "$raw_location_y",
                    "timestamp": "$timestamp"
                },
                **{
                    field_name: {
                        "$max": {
                            "$cond": [
                                {"$eq": ["$data.BSSID", bssid]},
                                "$data.RSSI",
                                None
                            ]
                        }
                    }
                    for bssid, field_name in ap_mapping.items()
                }
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    normalized_results = []
    for doc in results:
        raw_x = doc["_id"]["raw_location_x"]
        raw_y = doc["_id"]["raw_location_y"]
        
        norm_x, norm_y = normalize_picos_coordinates(
            raw_x, raw_y,
            origin_x if origin_x is not None else 0,
            origin_y if origin_y is not None else 0
        )
        
        if dry_run:
            new_doc = {
                "raw_location_x": raw_x,
                "raw_location_y": raw_y,
                "location_x": norm_x,
                "location_y": norm_y,
                "timestamp": doc["_id"]["timestamp"],
                **{field: doc.get(field) for field in ap_mapping.values()}
            }
        else:
            new_doc = {
                "location_x": norm_x,
                "location_y": norm_y,
                "timestamp": doc["_id"]["timestamp"],
                **{field: doc.get(field) for field in ap_mapping.values()}
            }
        normalized_results.append(new_doc)
    
    if dry_run:
        print(f"Dry run: Would process {len(normalized_results)} documents")
        print(f"Documentos would be processed into "f"{output_db.name}.{output_collection_name}")
        if normalized_results:
            print("Sample documents:")
            doc = normalized_results[0]
            print(f"  Raw location: {doc.get('raw_location_x')},{doc.get('raw_location_y')}")
            print(f"  Normalized location: {doc['location_x']:.4f},{doc['location_y']:.4f}")
            print(f"  timestamp: {datetime.fromtimestamp(doc['timestamp'])}")
            for ap in ap_mapping.values():
                print(f"  {ap}: {doc.get(ap, 'N/A')}")
            print()
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
        bssid: f"{ap_name}_rssi"
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
        output_db = client["wifi_fingerprinting_data"]

        # Collection name will match triangle name (e.g., reto_grande_wifi_client_data_global)
        output_collection = triangle_name

        # Run transform
        transform_wifi_data(
            db=input_db,
            origin_x=origin_x,
            origin_y=origin_y,
            start_time=start_time,
            end_time=end_time,
            dry_run=False,
            input_collection_name=input_collection,
            output_collection_name=output_collection,
            ap_mapping=flat_ap_mapping,
            output_db=output_db
        )