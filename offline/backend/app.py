from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Shared variable to store button ID
button_id = None

# Lock for thread safety
button_id_lock = threading.Lock()

def run_script_periodically():
    """Run the Python script every 15 seconds."""
    while True:
        with button_id_lock:
            current_button_id = button_id  # Access the shared variable

        if current_button_id is not None:
            command = ["python3", "run_script.py", str(current_button_id)]
            try:
                subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Script executed with button ID: {current_button_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error running script: {e.stderr}")
        
        time.sleep(15)  # Wait 15 seconds before running again

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['X-Content-Type-Options'] = '*'
        return res

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
    app.run(host='0.0.0.0', port=5050, debug=True)
