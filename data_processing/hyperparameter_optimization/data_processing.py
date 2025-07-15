from pymongo import MongoClient
import numpy as np

def get_dataset(collection_name, db_name):
    """Optimized data loading with proper type conversion and error handling"""
    client = MongoClient('mongodb://localhost:28910/', 
                       connectTimeoutMS=30000, 
                       socketTimeoutMS=30000,
                       maxPoolSize=20)
    
    db = client[db_name]
    collection = db[collection_name]
    
    # Use aggregation pipeline for server-side processing and type conversion
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
    
    # Get cursor with larger batch size
    cursor = collection.aggregate(pipeline, allowDiskUse=True, batchSize=50000)
    
    data = []
    for doc in cursor:
        try:
            # Convert all values to float explicitly
            row = (
                float(doc['AP1_rssi']),
                float(doc['AP2_rssi']),
                float(doc['AP3_rssi']),
                float(doc['location_x']),
                float(doc['location_y'])
            )
            data.append(row)
        except (ValueError, TypeError, KeyError) as e:
            # Skip malformed documents
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
