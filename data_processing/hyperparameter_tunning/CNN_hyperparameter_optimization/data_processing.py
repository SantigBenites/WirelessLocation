# data_processing.py
from typing import List, Sequence, Union
from pymongo import MongoClient
import numpy as np
from feature_lists import DATASET_TO_FEATURE

def get_feature_list(dataset:str) -> List[str]:
    """
    Accept either a preset name (str) or an explicit ordered list of feature keys.
    Returns a concrete list of feature names in the order they should be used.
    """
    if dataset in DATASET_TO_FEATURE:
        return DATASET_TO_FEATURE[dataset]
    raise ValueError(
        f"Unknown feature selection '{dataset}'. "
        f"Use one of {list(DATASET_TO_FEATURE.keys())} or pass a list of fields."
    )


def _default_value_for(feature_name: str) -> float:
    """
    Choose a safe numeric default for missing values.
    - RSSI-like quantities default to -100 dBm-ish.
    - Shares/ratios/power ratios and scalars default to 0.0.
    """
    name = feature_name.lower()
    if "_rssi" in name or "_rssi_1m" in name or "residual" in name:
        return -100.0
    return 0.0


from typing import List, Sequence, Union
from pymongo import MongoClient
import numpy as np
from feature_lists import DATASET_TO_FEATURE

def get_feature_list(dataset:str) -> List[str]:
    """
    Accept either a preset name (str) or an explicit ordered list of feature keys.
    Returns a concrete list of feature names in the order they should be used.
    """
    if dataset in DATASET_TO_FEATURE:
        return DATASET_TO_FEATURE[dataset]
    raise ValueError(
        f"Unknown feature selection '{dataset}'. "
        f"Use one of {list(DATASET_TO_FEATURE.keys())} or pass a list of fields."
    )


def _default_value_for(feature_name: str) -> float:
    """
    Choose a safe numeric default for missing values.
    - RSSI-like quantities default to -100 dBm-ish.
    - Shares/ratios/power ratios and scalars default to 0.0.
    """
    name = feature_name.lower()
    if "_rssi" in name or "_rssi_1m" in name or "residual" in name:
        return -100.0
    return 0.0


def get_dataset(
    collection_name: str,
    db_name: str,
    features: Union[str, Sequence[str]],
):
    """
    Load data from MongoDB and return a NumPy array where each row is:
        [ <features...>, location_x, location_y ]

    feature_selection: preset name or explicit list of fields (order = model input order)
    """

    client = MongoClient(
        "mongodb://localhost:28910/",
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        maxPoolSize=20,
    )
    db = client[db_name]
    collection = db[collection_name]

    # Build projection: computed fields via $ifNull to guarantee numeric values.
    projection = {
        "location_x": 1,
        "location_y": 1,
    }
    for f in features:
        projection[f] = {"$ifNull": [f"${f}", _default_value_for(f)]}

    # Keep only rows with numeric labels; features are numeric due to $ifNull above.
    pipeline = [
        {"$project": projection},
        {
            "$match": {
                "location_x": {"$type": "number"},
                "location_y": {"$type": "number"},
            }
        },
    ]

    cursor = collection.aggregate(pipeline, allowDiskUse=True, batchSize=50000)
    rows = []
    for doc in cursor:

        try:
            x = [float(doc[f]) for f in features]
            y = [float(doc["location_x"]), float(doc["location_y"])]
            rows.append(tuple(x + y))
        except Exception:
            # Skip malformed rows
            continue

    if not rows:
        raise ValueError(f"No valid data found in collection '{collection_name}' of DB '{db_name}'.")

    return np.array(rows, dtype=np.float32)


def split_combined_data(
    combined_array: np.ndarray,
    features: Union[str, Sequence[str]],
):
    """
    Split stacked array into (X, y) based on the selected feature list size.
    """
    n_features = len(features)
    X = combined_array[:, :n_features]
    y = combined_array[:, n_features:]  # [location_x, location_y]
    return X, y


def combine_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    return np.vstack(arrays)


def shuffle_array(arr: np.ndarray, random_state: int = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    idx = np.arange(arr.shape[0])
    rng.shuffle(idx)
    return arr[idx]
