#!/bin/bash

# Function to clean up processes on exit
cleanup() {
    echo "Stopping Flask backend and React frontend..."
    kill $FLASK_PID
    exit 0
}

# Trap SIGINT (Ctrl+C) signal
trap cleanup SIGINT

kill -9 $(lsof -t -i:5050)

echo "Starting Flask backend on port 5050..."
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5050 &
FLASK_PID=$!

# Wait for both processes to exit
wait $FLASK_PID