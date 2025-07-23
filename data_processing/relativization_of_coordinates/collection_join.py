from pymongo import MongoClient
from IPython.display import display, Markdown
from datetime import datetime

client = MongoClient("mongodb://localhost:28910/")
db = client["wifi_data_db"]


def union_collections_preserve_duplicates(collections_to_join, output_collection):
    
    # Verify input collections exist and get counts
    input_counts = {}
    for coll_name in collections_to_join:
        if coll_name not in db.list_collection_names():
            raise ValueError(f"Collection {coll_name} does not exist")
        input_counts[coll_name] = db[coll_name].count_documents({})
        print(f"Collection {coll_name} has {input_counts[coll_name]} documents")
    
    # Create a pipeline that adds source collection info and merges
    pipelines = []
    
    # First pipeline creates the output collection with modified _id
    first_coll = collections_to_join[0]
    pipelines.append([
        {"$addFields": {
            "original_id": "$_id",
            "_id": {"$concat": [first_coll, "||", {"$toString": "$_id"}]},
            "source_collection": first_coll
        }},
        {"$out": output_collection}
    ])
    
    # Subsequent pipelines merge with modified _id
    for coll_name in collections_to_join[1:]:
        pipelines.append([
            {"$addFields": {
                "original_id": "$_id",
                "_id": {"$concat": [coll_name, "||", {"$toString": "$_id"}]},
                "source_collection": coll_name
            }},
            {"$merge": {
                "into": output_collection,
                "whenMatched": "fail",  # Shouldn't happen with our new _id scheme
                "whenNotMatched": "insert"
            }}
        ])
    
    # Execute all pipelines
    for i, (coll_name, pipeline) in enumerate(zip(collections_to_join, pipelines)):
        db[coll_name].aggregate(pipeline)
        print(f"Processed {coll_name} ({i+1}/{len(collections_to_join)})")
    
    # Verify the output
    output_count = db[output_collection].count_documents({})
    expected_total = sum(input_counts.values())
    print(f"Output collection {output_collection} has {output_count} documents")
    print(f"Expected total (sum of inputs): {expected_total}")
    
    if output_count != expected_total:
        print(f"Warning: Output count doesn't match sum of input collections")
    
    return output_count

# Example usage
union_collections_preserve_duplicates(
    collections_to_join=["wifi_client_data", "wifi_client_data_1","wifi_client_data_2"],
    output_collection="wifi_client_data_global"
)