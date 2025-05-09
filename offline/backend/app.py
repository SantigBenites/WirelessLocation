from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time
from pymongo import MongoClient
from wifi_client_data import get_wifi_client_data, get_status

app = Flask(__name__)

# Shared variable to store button ID
button_id = None
is_script_running = False

# Lock for thread safety
button_id_lock = threading.Lock()
script_running_lock = threading.Lock()

# MongoDB configuration
MONGO_URI = "mongodb://127.0.0.1:28910/"
DATABASE_NAME = "wifi_data_db"
COLLECTION_NAME = "wifi_client_data"

# Initialize MongoDB client with error handling
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')  # Test the connection
    db = client[DATABASE_NAME]
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ Could not connect to MongoDB: {e}")
    db = None  # You'll need to handle this in your script logic

def run_script_periodically():
    """Run the Python script every 15 seconds and save results to MongoDB."""
    global is_script_running
    while True:

        with button_id_lock:
            current_button_id = button_id  # Access the shared variable

        if current_button_id is not None and is_script_running:
                print(current_button_id)
                # Get Wi-Fi client data
                wifi_data = get_wifi_client_data()

                # Save data to the time-series collection
                for pico_ip, data in wifi_data.items():
                    try:
                        if data != "error":
                            document = {
                                "metadata": {
                                    "button_id": current_button_id,
                                    "pico_ip": pico_ip
                                },
                                "data": data,
                                "timestamp": time.time()
                            }
                            db[COLLECTION_NAME].insert_one(document)
                            print(f"Data saved for {pico_ip} with button {current_button_id}")
                        else:
                            print(f"Error in {pico_ip}")
                    except Exception as e:
                        print(f"Error running script: {e}")

        time.sleep(0)  # Wait 15 seconds before running again

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
    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/')
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route('/update-button-id', methods=['POST'])
def update_button_id():
    """Update the button ID from the frontend."""
    global button_id, is_script_running
    data = request.get_json()
    new_button_id = data.get('buttonId')

    with button_id_lock:
        button_id = new_button_id

    with script_running_lock:
        is_script_running = True

    response = jsonify({"output": f"Button ID updated to {button_id}", "buttonId": button_id})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/stop-script', methods=['POST'])
def stop_script():
    """Stop the periodic script execution."""
    global is_script_running
    with script_running_lock:
        is_script_running = False
    
    response = jsonify({"output": "Script stopped"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Start the periodic task in a background thread
threading.Thread(target=run_script_periodically, daemon=True).start()

if __name__ == '__main__':
    CORS(app)  # Allow all origins
    app.run(host='0.0.0.0', port=5050, debug=True)