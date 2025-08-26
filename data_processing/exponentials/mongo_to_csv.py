
from pymongo import MongoClient
import pandas as pd

# connect

client = MongoClient("mongodb://localhost:28910/")
db = client["wifi_fingerprinting_data"]
collection = db["equilatero_grande_outdoor"]

# fetch all documents
cursor = collection.find({})

# convert to DataFrame
df = pd.DataFrame(list(cursor))

# optional: drop the MongoDB _id column if you don’t want it
df.drop(columns=["_id"], inplace=True)

# save to CSV
df.to_csv("users.csv", index=False)
