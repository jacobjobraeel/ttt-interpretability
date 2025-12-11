#!/bin/bash
set -e  # Exit on error

# Default mode is setup_only
MODE=${MODE:-setup_only}

echo "=== 1. Setting up Virtual Environment ==="
# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

echo "=== 2. Installing Dependencies ==="
pip install --upgrade pip
# Install requirements inside the venv
# Note: -f link is still good to have for jax, but we hardcoded jaxlib
pip install -r requirements/gpu_requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== 3. Downloading TTT Model Weights ==="
# Ensure the script is executable
chmod +x download_model.sh
./download_model.sh

if [ "$MODE" == "eval_wikitext" ]; then
    echo "=== 4. Running Inference Verification (Legacy: Wikitext + Public Llama 2) ==="
    # We use the public tokenizer to avoid login prompts
    # Enable allocator override to fix CUDNN_STATUS_INTERNAL_ERROR
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform python -m ttt.train \
        --mesh_dim='1,1,1' \
        --dtype="bf16" \
        --eval_mode=True \
        --dataset_name="wikitext" \
        --dataset_config_name="wikitext-103-v1" \
        --tokenizer_name="nousresearch/Llama-2-7b-hf" \
        --dataset_path="./data_cache" \
        --load_model_config="pickle::experiments/1b-TTT/metadata.pkl" \
        --load_part="trainstate" \
        --exp_dir="experiments" \
        --exp_name="1b-TTT" \
        --resume_exp_name="1b-TTT" \
        --seq_length=2048 \
        --global_batch_size=1
    echo "=== Legacy Eval Complete ==="

elif [ "$MODE" == "verify_synthetic" ]; then
    echo "=== 4. Running New Plasticity Verification (Synthetic) ==="
    ./scripts/analysis/experiments/verify_synthetic.sh
    echo "=== Verification Complete ==="

elif [ "$MODE" == "pg19_baseline" ]; then
    echo "=== 4. Running PG-19 Baseline ==="
    ./scripts/analysis/experiments/run_pg19_baseline.sh
    echo "=== PG-19 Baseline Complete ==="

else
    echo "=== Environment Setup Complete ==="
    echo "To run experiments, use one of:"
    echo "  MODE=verify_synthetic ./setup_and_run.sh"
    echo "  MODE=pg19_baseline ./setup_and_run.sh"
    echo "Or run the scripts directly from scripts/analysis/experiments/"
fi
