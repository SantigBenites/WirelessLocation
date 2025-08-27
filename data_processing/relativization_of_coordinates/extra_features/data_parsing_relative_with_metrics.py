
from pymongo import MongoClient
from datetime import datetime
from triangle_dict import triangle_dictionary, ap_mapping
from typing import Dict, Optional, Sequence, Tuple
from collections import defaultdict, Counter
import numpy as np
import math

def _pairwise_ratios(values: Dict[str, Optional[float]], cols: Sequence[str]) -> Dict[str, Optional[float]]:
    """
    Build {f"{a}_over_{b}": values[a]/values[b]} for all a != b in cols.
    Returns None when denominator is 0 or either value is None.
    """
    out: Dict[str, Optional[float]] = {}
    for a in cols:
        for b in cols:
            if a == b:
                continue
            va = values.get(a)
            vb = values.get(b)
            key = f"{a}_over_{b}"
            out[key] = (va / vb) if (va is not None and vb not in (None, 0)) else None
    return out
def _pairwise_power_ratios_from_dbm(values: Dict[str, Optional[float]], cols: Sequence[str]) -> Dict[str, Optional[float]]:
    """
    Compute physically meaningful power ratios by converting each dBm to linear power (mW)
    and then forming pairwise ratios. Keys are named "<a>_power_over_<b>".
    """
    # Convert to linear power (mW). dBm -> mW: 10 ** (dBm/10)
    linear = {}
    for k in cols:
        v = values.get(k)
        linear[k] = (10 ** (v / 10.0)) if v is not None else None
    # Reuse the generic ratio builder but rename keys
    raw = _pairwise_ratios(linear, cols)
    out = {}
    for k, val in raw.items():
        out[k.replace("_over_", "_power_over_")] = val
    return out


# -------------------------------
# Utility helpers
# -------------------------------

def normalize_picos_coordinates(x, y, origin_x, origin_y):
    pico_interval = 1 / 10
    ap_interval   = 1 / 4
    normalized_x = pico_interval * x
    normalized_y = pico_interval * y
    normalized_origin_x = ap_interval * origin_x
    normalized_origin_y = ap_interval * origin_y
    return (normalized_x - normalized_origin_x, normalized_y - normalized_origin_y)

def normalize_ap_positions(ap_positions: Dict[str, Tuple[float, float]], origin_x: float, origin_y: float):
    ap_interval = 1 / 4
    norm = {}
    for ap_name, (ax, ay) in ap_positions.items():
        norm[ap_name] = (ap_interval * ax - ap_interval * origin_x,
                         ap_interval * ay - ap_interval * origin_y)
    return norm

def calculate_centroid(point1, point2, point3):
    cx = (point1[0] + point2[0] + point3[0]) / 3
    cy = (point1[1] + point2[1] + point3[1]) / 3
    return (cx, cy)

def channel_to_freq_mhz(ch: int) -> Optional[float]:
    if ch is None:
        return None
    try:
        ch = int(ch)
    except Exception:
        return None
    if 1 <= ch <= 14:
        return 2412.0 + 5.0 * (ch - 1)
    if 32 <= ch <= 196:
        return 5000.0 + 5.0 * ch
    if 1 <= ch <= 233:
        return 5940.0 + 5.0 * ch
    return None

def safe_log10(x: float, eps: float = 1e-8) -> float:
    return math.log10(max(x, eps))

def softmax_shares(rssis: Sequence[Optional[float]]) -> Sequence[Optional[float]]:
    vals = [v for v in rssis if v is not None]
    if len(vals) == 0:
        return [None] * len(rssis)
    mx = max(vals)
    lin = [ (10 ** ((v - mx) / 10.0)) if v is not None else None for v in rssis ]
    denom = sum([x for x in lin if x is not None]) or 1.0
    return [ (x / denom) if x is not None else None for x in lin ]

# -------------------------------
# Core transformation
# -------------------------------

def transform_wifi_data(db,
                        origin_x=None,
                        origin_y=None,
                        ap_positions=None,
                        start_time=None,
                        end_time=None,
                        dry_run=False,
                        output_collection_name="wifi_data_filtered",
                        input_collection_name="wifi_data",
                        ap_mapping=None,
                        output_db=None,
                        debug=False):
    if ap_mapping is None:
        raise ValueError("ap_mapping must be provided (BSSID -> label)")
    if output_db is None:
        raise ValueError("output_db must be provided")
    if ap_positions is None:
        raise ValueError("ap_positions (raw) must be provided")

    ip_to_y = {
        31: 1, 32: 2, 33: 3, 34: 4, 35: 5,
        36: 6, 37: 7, 38: 8, 39: 9, 30: 10
    }

    match_stage = {}
    if start_time:
        match_stage["timestamp"] = {"$gte": start_time.timestamp()}
    if end_time:
        match_stage.setdefault("timestamp", {})["$lte"] = end_time.timestamp()

    collection = db[input_collection_name]
    normalized_ap_keys = set(ap_mapping.keys())
    ap_labels = sorted(set(ap_mapping.values()))

    raw_docs = list(collection.find(match_stage)) if match_stage else list(collection.find())

    cx, cy = origin_x, origin_y
    ap_pos_norm_raw_keys = normalize_ap_positions(ap_positions, cx, cy)
    apname_to_label = {ap_name: f"{ap_name}_rssi" for ap_name in ap_positions.keys()}
    label_to_ap_pos = {}
    for ap_name, (ax, ay) in ap_pos_norm_raw_keys.items():
        lbl = apname_to_label.get(ap_name)
        if lbl in ap_labels:
            label_to_ap_pos[lbl] = (ax, ay)

    label_sum = defaultdict(float)
    label_count = defaultdict(int)
    label_channel_counter = defaultdict(Counter)

    pre_docs = []
    for doc in raw_docs:
        if isinstance(doc.get("data"), dict) and "error" in doc["data"]:
            continue

        bssid_to_best = {}
        for entry in doc.get("data", []):
            bssid = str(entry.get("BSSID", "")).lower()
            rssi = entry.get("RSSI")
            ch = entry.get("Channel")
            if bssid in normalized_ap_keys and isinstance(rssi, (int, float)):
                prev = bssid_to_best.get(bssid)
                if (prev is None) or (rssi > prev["RSSI"]):
                    bssid_to_best[bssid] = {"RSSI": float(rssi), "Channel": ch}

        label_best = {}
        for bssid, label in ap_mapping.items():
            rec = bssid_to_best.get(bssid)
            if rec is not None:
                if (label not in label_best) or (rec["RSSI"] > label_best[label]["RSSI"]):
                    label_best[label] = {"RSSI": rec["RSSI"], "Channel": rec["Channel"]}

        for label, rec in label_best.items():
            label_sum[label] += rec["RSSI"]
            label_count[label] += 1
            if rec["Channel"] is not None:
                label_channel_counter[label][rec["Channel"]] += 1

        pre_docs.append((doc, label_best))

    label_avg = {label: (label_sum[label] / label_count[label]) for label in ap_labels if label_count[label] > 0}
    label_mode_channel = {}
    for label in ap_labels:
        if label_channel_counter[label]:
            mode_ch, _ = label_channel_counter[label].most_common(1)[0]
            label_mode_channel[label] = mode_ch

    if debug:
        print("Triangle averages per AP label:", {k: round(v, 2) for k, v in label_avg.items()})
        print("Modal channel per AP label:", label_mode_channel)

    X_rows = []
    y_vals = []

    def doc_to_norm_xy(d):
        pico_ip = d["metadata"]["pico_ip"]
        ip_ending = int(pico_ip.split(".")[3])
        raw_y = ip_to_y.get(ip_ending)
        raw_x = d["metadata"]["button_id"]
        if raw_y is None:
            return None
        nx, ny = normalize_picos_coordinates(raw_x, raw_y, cx, cy)
        return nx, ny

    for d, label_best in pre_docs:
        loc = doc_to_norm_xy(d)
        if loc is None:
            continue
        nx, ny = loc
        for label, rec in label_best.items():
            ap_xy = label_to_ap_pos.get(label)
            if ap_xy is None:
                continue
            ch = rec.get("Channel")
            f_mhz = channel_to_freq_mhz(ch)
            if f_mhz is None:
                ch = label_mode_channel.get(label)
                f_mhz = channel_to_freq_mhz(ch)
                if f_mhz is None:
                    continue

            dx = nx - ap_xy[0]
            dy = ny - ap_xy[1]
            d = math.sqrt(dx*dx + dy*dy)
            logd = safe_log10(d)
            fterm = 20.0 * safe_log10(f_mhz)

            onehots = [1.0 if label == l else 0.0 for l in ap_labels]
            X_rows.append(onehots + [logd, fterm])
            y_vals.append(rec["RSSI"])

    beta = None
    if len(X_rows) >= (len(ap_labels) + 2):
        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        beta = np.zeros(len(ap_labels) + 2, dtype=float)
        for i, l in enumerate(ap_labels):
            beta[i] = label_avg.get(l, -60.0)
        beta[-2] = -20.0
        beta[-1] = 0.0

    ap_intercepts = {l: float(beta[i]) for i, l in enumerate(ap_labels)}
    beta1 = float(beta[-2])
    beta2 = float(beta[-1])
    n_est = -beta1 / 10.0 if beta1 != 0 else None

    if debug:
        print("LDPL fit: ap_intercepts:", {k: round(v, 2) for k, v in ap_intercepts.items()})
        print(f"LDPL fit: beta1 (log10 d)={beta1:.3f}, beta2 (20log10 f)={beta2:.3f}, n_est≈{n_est:.3f}")

    normalized_results = []
    for d, label_best in pre_docs:
        try:
            pico_ip = d["metadata"]["pico_ip"]
            ip_ending = int(pico_ip.split(".")[3])
            raw_y = ip_to_y.get(ip_ending)
            raw_x = d["metadata"]["button_id"]
            timestamp = d["timestamp"]
            if raw_y is None:
                continue

            nx, ny = normalize_picos_coordinates(raw_x, raw_y, cx, cy)

            ap_rssi = {label: None for label in ap_labels}
            ap_chan = {label: None for label in ap_labels}
            for label, rec in label_best.items():
                ap_rssi[label] = float(rec["RSSI"])
                ap_chan[label] = rec.get("Channel")

            for label in ap_labels:
                if ap_rssi[label] is None and label in label_avg:
                    ap_rssi[label] = float(int(round(label_avg[label])))
                if ap_chan[label] is None and label in label_mode_channel:
                    ap_chan[label] = label_mode_channel[label]

            residuals = {}
            rssi_1m = {}
            for label in ap_labels:
                rssi = ap_rssi.get(label)
                ch = ap_chan.get(label)
                if rssi is None:
                    residuals[label] = None
                    rssi_1m[label] = None
                    continue

                ap_xy = label_to_ap_pos.get(label)
                if ap_xy is None:
                    residuals[label] = None
                    rssi_1m[label] = None
                    continue

                dx = nx - ap_xy[0]
                dy = ny - ap_xy[1]
                dist = math.sqrt(dx*dx + dy*dy)
                logd = safe_log10(dist)
                f_mhz = channel_to_freq_mhz(ch)
                fterm = 20.0 * safe_log10(f_mhz) if f_mhz else 0.0

                pred = ap_intercepts[label] + beta1 * logd + beta2 * fterm
                residuals[label] = rssi - pred
                rssi_1m[label] = rssi - beta1 * logd

            def bias_correct(v, lbl):
                if v is None:
                    return None
                return v - ap_intercepts.get(lbl, 0.0)

            bc = [bias_correct(ap_rssi[l], l) for l in ap_labels]

            def delta(a, b):
                return (a - b) if (a is not None and b is not None) else None

            deltas = {}
            if len(ap_labels) >= 3:
                lA, lB, lC = ap_labels[:3]
                deltas[f"delta_{lA}_{lB}"] = delta(bc[0], bc[1])
                deltas[f"delta_{lB}_{lC}"] = delta(bc[1], bc[2])
                deltas[f"delta_{lC}_{lA}"] = delta(bc[2], bc[0])

            shares_vals = softmax_shares([ap_rssi[l] for l in ap_labels])
            shares = {f"{lbl}_share": shares_vals[i] for i, lbl in enumerate(ap_labels)}

            ratio_cols = _pairwise_ratios(ap_rssi, list(ap_labels))
            power_ratio_cols = _pairwise_power_ratios_from_dbm(ap_rssi, list(ap_labels))


            new_doc = {
                "location_x": nx,
                "location_y": ny,
                "timestamp": timestamp,
                **{lbl: ap_rssi[lbl] for lbl in ap_labels},
                **deltas,
                **shares,
                **ratio_cols,
                **power_ratio_cols,
                **{f"{lbl}_residual": residuals[lbl] for lbl in ap_labels},
                **{f"{lbl}_rssi_1m": rssi_1m[lbl] for lbl in ap_labels},
                **{f"ap_intercepts_{lbl}": ap_intercepts[lbl] for lbl in ap_labels},
                "beta1_log10d": beta1,
                "beta2_20log10f": beta2,
                "n_est": n_est,
            }

            normalized_results.append(new_doc)

        except Exception as e:
            print(f"⚠️ Error processing document: {e}")
            continue

    if dry_run:
        print(f"Dry run: Would process {len(normalized_results)} documents")
        print(f"Documents would be processed into {output_db.name}.{output_collection_name}")
        if normalized_results:
            doc = normalized_results[0]
            print(f"  Normalized location: {doc['location_x']:.4f},{doc['location_y']:.4f}")
            print(f"  timestamp: {datetime.fromtimestamp(doc['timestamp'])}")
            for lbl in ap_labels:
                print(f"  {lbl}: {doc.get(lbl, 'N/A')}, share={doc.get(f'{lbl}_share')}")
        return normalized_results

    if normalized_results:
        output_db[output_collection_name].delete_many({})
        output_db[output_collection_name].insert_many(normalized_results)
        print(f"✅ Processed {len(normalized_results)} documents into "
              f"{output_db.name}.{output_collection_name}")
        return normalized_results
    else:
        print("No documents matched the criteria")
        return []


if __name__ == "__main__":

    # Connect to your MongoDB
    client = MongoClient("mongodb://localhost:28910/")  # Update as needed

    # Flatten the ap_mapping: BSSID → label (e.g., "freind1_rssi")
    flat_ap_mapping = {
        bssid.lower(): f"{ap_name}_rssi"
        for ap_name, bssids in ap_mapping.items()
        for bssid in bssids
    }

    # Iterate through each triangle setup
    for triangle_name, current_triangle in triangle_dictionary.items():
        input_db_name       = current_triangle["db"]
        input_collection    = current_triangle["collection"]
        start_time          = current_triangle["start"]
        end_time            = current_triangle["end"]
        ap_positions        = current_triangle["ap_positions"]  # raw positions in grid units

        # Calculate centroid of the triangle for normalization
        origin_x, origin_y = calculate_centroid(
            *[ap_positions[ap] for ap in ["freind1", "freind2", "freind3"]]
        )

        # Select input DB
        input_db = client[input_db_name]

        # Output DB is always this
        output_db = client["wifi_fingerprinting_data_extra_features"]

        # Collection name will match triangle name (e.g., reto_grande_wifi_client_data_global)
        output_collection = triangle_name

        # Run transform
        transform_wifi_data(
            db=input_db,
            origin_x=origin_x,
            origin_y=origin_y,
            ap_positions=ap_positions,
            start_time=start_time,
            end_time=end_time,
            dry_run=False,
            input_collection_name=input_collection,
            output_collection_name=output_collection,
            ap_mapping=flat_ap_mapping,
            output_db=output_db,
            debug=False
        )