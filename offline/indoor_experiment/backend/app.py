from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time
from pymongo import MongoClient
from wifi_client_data import get_wifi_client_data, get_status

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# MongoDB setup
MONGO_URI = "mongodb://127.0.0.1:28910/"
DATABASE_NAME = "wifi_data_db_indoor"
COLLECTION_NAME = "wifi_indoor_data_2"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[DATABASE_NAME]
    print("✅ MongoDB connected")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# Global mappings and script control
coordinates_map = {}  # { '192.168.1.31': {'x': 1, 'y': 2} }
active_pico_ips = set()
is_script_running = False  # Default: script not running
script_running_lock = threading.Lock()

def run_script_periodically():
    while True:
        if is_script_running and db is not None:
            print(active_pico_ips)
            wifi_data = get_wifi_client_data(active_pico_ips)
            if wifi_data != None:
                for pico_ip, data in wifi_data.items():
                    coords = coordinates_map.get(pico_ip, {})

                    # Error format handling
                    if isinstance(data, dict) and "error" in data:
                        print(f"❌ Error from {pico_ip}: {data['error']}")
                        continue

                    # Valid scan data is a list of BSSIDs
                    if isinstance(data, list) and coords:
                        document = {
                            "metadata": {
                                "x": coords.get("x"),
                                "y": coords.get("y"),
                                "pico_ip": pico_ip,
                                "triangle_shape": coords.get("triangle_shape")
                            },
                            "data": data,
                            "timestamp": time.time()
                        }
                        #print(document)
                        db[COLLECTION_NAME].insert_one(document)
                        print(f"✅ Saved scan data from {pico_ip} at {coords}")
                    else:
                        print(f"⚠️ Invalid data format from {pico_ip}: {type(data)}")
        time.sleep(2)

@app.route('/')
def health():
    global coordinates_map
    return jsonify({
        "status": "running" if is_script_running else "stopped",
        "coordinates": coordinates_map
    })

@app.route('/check-status', methods=['GET'])
def check_status():
    status = get_status()
    return jsonify(status)

@app.route('/update-coordinates', methods=['POST'])
def update_coordinates():
    global coordinates_map
    data = request.get_json()
    triangle_shape = data.pop("triangle_shape", None)
    print(triangle_shape)
    coordinates_map = data

    if triangle_shape:
        for ip in coordinates_map:
            coordinates_map[ip]["triangle_shape"] = triangle_shape

    return jsonify({"status": "Coordinates updated", "data": coordinates_map})

@app.route('/update-active-picos', methods=['POST'])
def update_active_picos():
    global active_pico_ips
    data = request.get_json()
    active_pico_ips = set(data.get("ips", []))
    return jsonify({"status": "Active Pico list updated", "active_ips": list(active_pico_ips)})

@app.route('/stop-script', methods=['POST'])
def stop_script():
    global is_script_running
    with script_running_lock:
        is_script_running = False
    return jsonify({"status": "Script stopped"})

@app.route('/start-script', methods=['POST'])
def start_script():
    global is_script_running
    with script_running_lock:
        is_script_running = True
    return jsonify({"status": "Script started"})

threading.Thread(target=run_script_periodically, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
