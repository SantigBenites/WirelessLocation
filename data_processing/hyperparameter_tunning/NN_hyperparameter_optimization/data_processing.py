from pymongo import MongoClient
import numpy as np

# data_processing.py

from pymongo import MongoClient
import numpy as np

def get_dataset(collection_name, db_name, feature_mode):
    """
    Load dataset in the new format and choose features:
      - feature_mode="rssi"   -> freind1/2/3_rssi (3 features)
      - feature_mode="ratios" -> 6 ratio features
      - feature_mode="both"   -> 3 RSSI + 6 ratios (9 features)
    Labels remain location_x, location_y.
    """
    feature_mode = feature_mode.lower()
    assert feature_mode in {"rssi", "ratios", "both"}, "feature_mode must be 'rssi', 'ratios', or 'both'"

    client = MongoClient('mongodb://localhost:28910/',
                         connectTimeoutMS=30000,
                         socketTimeoutMS=30000,
                         maxPoolSize=20)

    db = client[db_name]
    collection = db[collection_name]

    # Base fields (labels)
    projection = {
        'location_x': 1,
        'location_y': 1,
    }

    # RSSI features
    if feature_mode in {"rssi", "both"}:
        projection.update({
            'freind1_rssi': {'$ifNull': ['$freind1_rssi', -100]},
            'freind2_rssi': {'$ifNull': ['$freind2_rssi', -100]},
            'freind3_rssi': {'$ifNull': ['$freind3_rssi', -100]},
        })

    # Ratio features
    if feature_mode in {"ratios", "both"}:
        projection.update({
            'freind1_rssi_over_freind2_rssi': {'$ifNull': ['$freind1_rssi_over_freind2_rssi', 0]},
            'freind1_rssi_over_freind3_rssi': {'$ifNull': ['$freind1_rssi_over_freind3_rssi', 0]},
            'freind2_rssi_over_freind1_rssi': {'$ifNull': ['$freind2_rssi_over_freind1_rssi', 0]},
            'freind2_rssi_over_freind3_rssi': {'$ifNull': ['$freind2_rssi_over_freind3_rssi', 0]},
            'freind3_rssi_over_freind1_rssi': {'$ifNull': ['$freind3_rssi_over_freind1_rssi', 0]},
            'freind3_rssi_over_freind2_rssi': {'$ifNull': ['$freind3_rssi_over_freind2_rssi', 0]},
        })

    # Match: require labels and whichever features we selected to be numeric
    match_stage = {
        'location_x': {'$type': 'number'},
        'location_y': {'$type': 'number'},
    }
    if feature_mode in {"rssi", "both"}:
        match_stage.update({
            'freind1_rssi': {'$type': 'number'},
            'freind2_rssi': {'$type': 'number'},
            'freind3_rssi': {'$type': 'number'},
        })
    if feature_mode in {"ratios", "both"}:
        # ratios can be 0 if missing (filled by $ifNull), still numeric
        match_stage.update({
            'freind1_rssi_over_freind2_rssi': {'$type': 'number'},
            'freind1_rssi_over_freind3_rssi': {'$type': 'number'},
            'freind2_rssi_over_freind1_rssi': {'$type': 'number'},
            'freind2_rssi_over_freind3_rssi': {'$type': 'number'},
            'freind3_rssi_over_freind1_rssi': {'$type': 'number'},
            'freind3_rssi_over_freind2_rssi': {'$type': 'number'},
        })

    pipeline = [
        {'$project': projection},
        {'$match': match_stage},
    ]

    cursor = collection.aggregate(pipeline, allowDiskUse=True, batchSize=50000)

    data = []
    for doc in cursor:
        try:
            row = []
            if feature_mode in {"rssi", "both"}:
                row.extend([
                    float(doc['freind1_rssi']),
                    float(doc['freind2_rssi']),
                    float(doc['freind3_rssi']),
                ])
            if feature_mode in {"ratios", "both"}:
                row.extend([
                    float(doc['freind1_rssi_over_freind2_rssi']),
                    float(doc['freind1_rssi_over_freind3_rssi']),
                    float(doc['freind2_rssi_over_freind1_rssi']),
                    float(doc['freind2_rssi_over_freind3_rssi']),
                    float(doc['freind3_rssi_over_freind1_rssi']),
                    float(doc['freind3_rssi_over_freind2_rssi']),
                ])
            row.extend([float(doc['location_x']), float(doc['location_y'])])
            data.append(tuple(row))
        except Exception:
            # skip malformed
            continue

    if not data:
        raise ValueError(f"No valid data found in collection {collection_name}")

    return np.array(data, dtype=np.float32)


def split_combined_data(combined_array, feature_mode):
    feature_mode = feature_mode.lower()
    assert feature_mode in {"rssi", "ratios", "both"}

    base_rssi = 3
    ratio_feats = 6
    if feature_mode == "rssi":
        num_features = base_rssi
    elif feature_mode == "ratios":
        num_features = ratio_feats
    else:  # both
        num_features = base_rssi + ratio_feats

    features = combined_array[:, :num_features]
    labels = combined_array[:, num_features:]
    return features, labels

def combine_arrays(arrays):
    return np.vstack(arrays)

def shuffle_array(arr, random_state=None):
    np.random.seed(random_state)
    shuffled_arr = arr.copy()
    np.random.shuffle(shuffled_arr)
    return shuffled_arr
