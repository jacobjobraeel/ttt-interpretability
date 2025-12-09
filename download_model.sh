#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Directory where train.py expects experiments to be
TARGET_DIR="experiments/1b-TTT"

echo "Creating directory $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

# Base URL for the Hugging Face repository
REPO_URL="https://huggingface.co/Test-Time-Training/ttt-linear-1.3b-books-32k/resolve/main"

echo "Downloading Checkpoint files to $TARGET_DIR..."
echo "This may take a minute or two depending on network speed..."

# 1. Download the main weights (approx 7.7GB)
# We use wget with -c to allow resuming if the connection drops
echo "Downloading streaming_train_state..."
wget -c -O "$TARGET_DIR/streaming_train_state" "$REPO_URL/streaming_train_state?download=true"

# 2. Download metadata (Config info)
echo "Downloading metadata.pkl..."
wget -c -O "$TARGET_DIR/metadata.pkl" "$REPO_URL/metadata.pkl?download=true"

# 3. Download dataset state (Optional, but good for reproducibility)
echo "Downloading dataset.pkl..."
wget -c -O "$TARGET_DIR/dataset.pkl" "$REPO_URL/dataset.pkl?download=true"

echo "-------------------------------------------------------"
echo "Download Complete!"
echo "Model files are located in: $(pwd)/$TARGET_DIR"
echo "You can now run inference pointing exp_dir to 'experiments' and resume_exp_name to '1b-TTT'"
