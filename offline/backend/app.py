from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['X-Content-Type-Options'] = '*'
        return res

@app.route('/')
def health_check():
    # Return a simple health check message with a 200 status code
    return jsonify({"status": "OK"}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.get_json()
    button_id = data.get('buttonId')
    timestamp = data.get('timestamp')

    command = ["python3", "run_script.py", str(button_id), timestamp]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return jsonify({"output": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr}), 500

if __name__ == '__main__':
    app.run(port=5050)
    app.debug = True
