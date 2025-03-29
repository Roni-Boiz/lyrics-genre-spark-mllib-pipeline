#!/bin/bash

# Detect OS
if [ "$OSTYPE" == "linux-gnu" ]; then
    OS="Linux"
elif [ "$OSTYPE" == "msys" ] || [ "$OSTYPE" == "cygwin" ]; then
    OS="Windows"
else
    echo "Unsupported OS detected!"
    exit 1
fi

echo "Detected OS: $OS"

if [ "$OS" == "Linux" ] || [ "$OS" == "Windows" ]; then

    echo "Starting frontend..."
    cd frontend || exit
    npm install
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    sleep 1

    echo "Starting backend..."
    cd backend || exit
    "$PYSPARK_PYTHON" -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python main.py  
    BACKEND_PID=$!
    cd ..
    
    echo "Stopping frontend and backend..."
    kill $FRONTEND_PID
    kill $BACKEND_PID
    
    exit 1
fi


