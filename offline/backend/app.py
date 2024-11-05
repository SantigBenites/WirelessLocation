from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

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
