from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb://localhost:28910/")
db = client["wifi_data_db"]
collection = db["wifi_client_data_global"]

# Aggregation to count valid (button_id, pico_ip) entries
pipeline = [
    {"$match": {
        "metadata.pico_ip": {"$exists": True},
        "metadata.button_id": {"$exists": True},
        "data.error": {"$exists": False}  # 🧼 Exclude documents with error field
    }},
    {"$group": {
        "_id": {
            "pico_ip": "$metadata.pico_ip",
            "button_id": "$metadata.button_id"
        },
        "count": {"$sum": 1}
    }}
]

results = list(collection.aggregate(pipeline))

# Prepare data for pivot table
data = [(doc["_id"]["button_id"], doc["_id"]["pico_ip"], doc["count"]) for doc in results]
df = pd.DataFrame(data, columns=["button_id", "pico_ip", "count"])
matrix = df.pivot(index="button_id", columns="pico_ip", values="count").fillna(0).astype(int)

# Add totals
matrix["Row_Total"] = matrix.sum(axis=1)
totals_row = matrix.sum(axis=0).to_frame().T
totals_row.index = ["Column_Total"]
matrix_with_totals = pd.concat([matrix, totals_row])

# Print the matrix with totals
print("\n📊 Document Count Matrix (filtered, with totals):\n")
print(matrix_with_totals.to_string())

# Optional: Export
# matrix_with_totals.to_csv("button_ip_matrix_filtered.csv")
