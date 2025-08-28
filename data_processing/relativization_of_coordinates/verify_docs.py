from pymongo import MongoClient
from triangle_dict import triangle_dictionary, triangle_dictionary_indoor

client = MongoClient("mongodb://localhost:28910/")

total_expected = 0
total_processed = 0

print("🔍 Verifying document migration (including indoor and outdoor):\n")
dst_db_name = "wifi_fingerprinting_data_extra_features"

def verify_entry(triangle_name, triangle_info, is_indoor=False):
    global total_expected, total_processed

    src_db_name = triangle_info["db"]
    src_collection_name = triangle_info["collection"]
    dst_collection_name = triangle_name

    src_db = client[src_db_name]
    dst_db = client[dst_db_name]

    if is_indoor:
        # Match based on triangle shape in metadata
        triangle_shape = triangle_info["triangle_name"]
        query = {"metadata.triangle_shape": triangle_shape}
    else:
        # Match based on timestamp
        start_ts = triangle_info["start"].timestamp()
        end_ts = triangle_info["end"].timestamp()
        query = {
            "timestamp": {
                "$gte": start_ts,
                "$lte": end_ts
            }
        }

    src_count = src_db[src_collection_name].count_documents(query)
    dst_count = dst_db[dst_collection_name].count_documents({})

    total_expected += src_count
    total_processed += dst_count

    status = "✅ MATCH" if src_count == dst_count else "❌ MISMATCH"
    source_label = f"Source: {src_count:<6}"
    target_label = f"→ Target: {dst_count:<6}"
    flag = "🟩 Indoor" if is_indoor else "🟦 Outdoor"

    print(f"{triangle_name:<35} | {source_label} {target_label} | {status} {flag}")


# Verify all outdoor and garage triangles
for triangle_name, triangle_info in triangle_dictionary.items():
    verify_entry(triangle_name, triangle_info, is_indoor=False)

# Verify all indoor triangles
for triangle_name, triangle_info in triangle_dictionary_indoor.items():
    verify_entry(triangle_name, triangle_info, is_indoor=True)

# Summary
print("\n📊 Summary:")
print(f"Destination db {dst_db_name}")
print(f"Total expected (source):  {total_expected}")
print(f"Total processed (target): {total_processed}")
print(f"{'✅ All matched!' if total_expected == total_processed else '⚠️  Some data is missing or extra'}")
