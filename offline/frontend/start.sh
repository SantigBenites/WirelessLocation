#!/bin/bash

export REACT_APP_NO_CLEAR_CONSOLE=true 
export BROWSER=none
export NODE_OPTIONS=--openssl-legacy-provider

# Function to clean up processes on exit
cleanup() {
    echo "Stopping Flask backend and React frontend..."
    kill $FLASK_PID
    kill $NPM_PID
    exit 0
}

# Trap SIGINT (Ctrl+C) signal
trap cleanup SIGINT

# Navigate to frontend and start React app
echo "Starting React frontend..."
npm start
NPM_PID=$!