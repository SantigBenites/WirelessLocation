from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time
from pymongo import MongoClient
from wifi_client_data import get_wifi_client_data, get_status

app = Flask(__name__)

# Shared variable to store button ID
button_id = None

# Lock for thread safety
button_id_lock = threading.Lock()

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "wifi_data_db"
COLLECTION_NAME = "wifi_client_data"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

def run_script_periodically():
    """Run the Python script every 15 seconds and save results to MongoDB."""
    while True:
        with button_id_lock:
            current_button_id = button_id  # Access the shared variable

        if current_button_id is not None:
            try:
                # Get Wi-Fi client data
                wifi_data = get_wifi_client_data()

                # Save data to the time-series collection
                for pico_ip, data in wifi_data.items():
                    document = {
                        "metadata": {
                            "button_id": current_button_id,
                            "pico_ip": pico_ip
                        },
                        "data": data,
                        "timestamp": time.time()
                    }
                    db[COLLECTION_NAME].insert_one(document)

                print(f"Data saved to MongoDB for button ID: {current_button_id}")
            except Exception as e:
                print(f"Error running script: {e}")

        time.sleep(15)  # Wait 15 seconds before running again

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'  # Allow all origins
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return res

@app.route('/check-status', methods=['GET'])
def check_status():
    """Check the status of all devices."""
    results = get_status()
    return jsonify(results)

@app.route('/')
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route('/update-button-id', methods=['POST'])
def update_button_id():
    """Update the button ID from the frontend."""
    global button_id
    data = request.get_json()
    new_button_id = data.get('buttonId')

    with button_id_lock:
        button_id = new_button_id

    return jsonify({"output": f"Button ID updated to {button_id}", "buttonId": button_id})

# Start the periodic task in a background thread
threading.Thread(target=run_script_periodically, daemon=True).start()

if __name__ == '__main__':
    CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
    app.use(CORS()) 
    app.run(host='0.0.0.0', port=5050, debug=True)