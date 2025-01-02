#!/bin/bash

# Exit script on any error
set -e

# Define paths
VENV_DIR="./venv"
REQUIREMENTS_FILE="requirements.txt"
DOWNLOAD_SCRIPT="download_models.py"

# Check if Python 3.11 is available
if ! python3 --version | grep -q "3.11"; then
    echo "Python 3.11 is not installed. Please install it first."
    exit 1
fi

# Step 1: Create Virtual Environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Deleting it..."
    rm -rf "$VENV_DIR"
fi

echo "Creating a new virtual environment..."
python3 -m venv "$VENV_DIR"

# Step 2: Activate Virtual Environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 3: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install Requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing requirements..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Error: $REQUIREMENTS_FILE not found."
    deactivate
    exit 1
fi

# Step 5: Download Models
if [ -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Downloading models..."
    python "$DOWNLOAD_SCRIPT"
else
    echo "Error: $DOWNLOAD_SCRIPT not found."
    deactivate
    exit 1
fi

# Step 6: Completion Message
echo "Setup complete! Virtual environment created, requirements installed, and models downloaded."

