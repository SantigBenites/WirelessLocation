
from pymongo import MongoClient
import numpy as np

GRID_SIZE = 32

def get_dataset(collection_name, db_name):
    client = MongoClient('mongodb://localhost:28910/', connectTimeoutMS=30000, socketTimeoutMS=30000, maxPoolSize=20)
    db = client[db_name]
    collection = db[collection_name]
    pipeline = [
        {'$project': {
            'AP1_rssi': {'$ifNull': ['$AP1_rssi', -100]},
            'AP2_rssi': {'$ifNull': ['$AP2_rssi', -100]},
            'AP3_rssi': {'$ifNull': ['$AP3_rssi', -100]},
            'location_x': 1,
            'location_y': 1
        }},
        {'$match': {
            'location_x': {'$type': 'number'},
            'location_y': {'$type': 'number'},
            'AP1_rssi': {'$type': 'number'},
            'AP2_rssi': {'$type': 'number'},
            'AP3_rssi': {'$type': 'number'}
        }}
    ]
    cursor = collection.aggregate(pipeline, allowDiskUse=True, batchSize=50000)
    data = []
    for doc in cursor:
        try:
            row = (
                float(doc['AP1_rssi']),
                float(doc['AP2_rssi']),
                float(doc['AP3_rssi']),
                float(doc['location_x']),
                float(doc['location_y'])
            )
            data.append(row)
        except:
            continue
    if not data:
        raise ValueError(f"No valid data found in collection {collection_name}")
    return np.array(data, dtype=np.float32)

def split_combined_data(combined_array, num_ap=3):
    features = combined_array[:, :num_ap]
    labels = combined_array[:, num_ap:]
    return features, labels

def combine_arrays(arrays):
    return np.vstack(arrays)

def shuffle_array(arr, random_state=None):
    np.random.seed(random_state)
    shuffled_arr = arr.copy()
    np.random.shuffle(shuffled_arr)
    return shuffled_arr

def features_to_sparse_grid(features, locations, grid_size=GRID_SIZE):
    num_samples = features.shape[0]
    num_channels = features.shape[1]
    grids = np.zeros((num_samples, num_channels, grid_size, grid_size), dtype=np.float32)
    for i in range(num_samples):
        gx = int(((locations[i,0] + 2) / 4) * (grid_size - 1))
        gy = int(((locations[i,1] + 2) / 4) * (grid_size - 1))
        grids[i, :, gy, gx] = features[i]
    return grids

