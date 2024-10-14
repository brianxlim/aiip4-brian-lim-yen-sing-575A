#!/bin/bash

echo "Starting the machine learning pipeline..."

echo "Activating virtual environment"
python3 -m venv .venv
source .venv/bin/activate

# Run the main Python script
echo "Running the machine learning pipeline..."
export PYTHONPATH=$(pwd)
python3 src/main.py

# Signal the end of the process
echo "Machine learning pipeline execution completed."
