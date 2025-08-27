from typing import Dict, Any
from pymongo import MongoClient

def write_results(client: MongoClient, db_name: str, collection: str, doc: Dict[str, Any]) -> None:
    try:
        db = client[db_name]
        db[collection].insert_one(doc)
        print(f"Results inserted into MongoDB collection '{collection}'.")
    except Exception as e:
        print("Failed to write results back to MongoDB:", e)