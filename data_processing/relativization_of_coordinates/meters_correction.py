#!/usr/bin/env python3
"""
Read docs from a MongoDB source collection, scale location_x/location_y by 3.2,
and upsert the transformed docs into a NEW destination collection.

No CLI arguments. Configure in the CONFIG section below.
"""

from pymongo import MongoClient, ReplaceOne

# --------------------------- CONFIG ---------------------------
MONGO_URI = "mongodb://localhost:28910/"
DB_NAME = "wifi_fingerprinting_data_raw"
SCALE_FACTOR = 3.2
BATCH_SIZE = 1000
QUERY = {}           # optional filter, e.g. {"location_x": {"$exists": True}}
COPY_INDEXES = False # set True to copy indexes from source to destination
# --------------------------------------------------------------

all_collections = [
    "equilatero_grande_garage",
    "equilatero_grande_outdoor",
    "equilatero_medio_garage",
    "equilatero_medio_outdoor",
    "isosceles_grande_indoor",
    "isosceles_grande_outdoor",
    "isosceles_medio_outdoor",
    "obtusangulo_grande_outdoor",
    "obtusangulo_pequeno_outdoor",
    "reto_grande_garage",
    "reto_grande_indoor",
    "reto_grande_outdoor",
    "reto_medio_garage",
    "reto_medio_outdoor",
    "reto_n_quadrado_grande_indoor",
    "reto_n_quadrado_grande_outdoor",
    "reto_n_quadrado_pequeno_outdoor",
    "reto_pequeno_garage",
    "reto_pequeno_outdoor",
]

def scale_xy(doc, factor=SCALE_FACTOR):
    out = dict(doc)  # copy so we don't mutate the original
    for k in ("location_x", "location_y"):
        v = out.get(k, None)
        if isinstance(v, (int, float)):
            out[k] = v * factor
    return out

def copy_indexes(src_coll, dst_coll):
    """
    Copy indexes (except _id_) from src_coll to dst_coll.
    Safe to run multiple times.
    """
    info = src_coll.index_information()
    for name, meta in info.items():
        if name == "_id_":
            continue
        keys = meta["key"]  # list of (field, direction)
        # Build options (exclude 'key' and 'v' which are internal)
        opts = {k: v for k, v in meta.items() if k not in {"key", "v"}}
        dst_coll.create_index(keys, **opts)

def process_colletion(collection_name):
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # fail fast if cannot connect
    db = client[DB_NAME]
    src = db[collection_name]
    db = client["wifi_fingerprinting_data_meters"]
    dst = db[collection_name]

    if COPY_INDEXES:
        copy_indexes(src, dst)

    cursor = src.find(QUERY, no_cursor_timeout=True).batch_size(BATCH_SIZE)

    ops = []
    processed = 0

    try:
        for doc in cursor:
            new_doc = scale_xy(doc)
            ops.append(ReplaceOne({"_id": doc["_id"]}, new_doc, upsert=True))
            if len(ops) >= BATCH_SIZE:
                result = dst.bulk_write(ops, ordered=False)
                processed += len(ops)
                print(f"Upserted batch of {len(ops)}; total so far: {processed}")
                ops = []
        if ops:
            result = dst.bulk_write(ops, ordered=False)
            processed += len(ops)
            print(f"Upserted final batch of {len(ops)}; total: {processed}")
    finally:
        cursor.close()

    print(f"Done. Upserted {processed} documents into '{DB_NAME}.{collection_name}'.")

if __name__ == "__main__":
    for collection in all_collections:
        process_colletion(collection)
