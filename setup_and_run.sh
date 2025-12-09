#!/bin/bash
set -e  # Exit on error

echo "=== 1. Installing Dependencies ==="
pip install --upgrade pip
# Install requirements. Note: Lambda usually has CUDA, so we trust the requirements file.
pip install -r requirements/gpu_requirements.txt
# Install HuggingFace Hub for downloading
pip install huggingface_hub

echo "=== 2. Downloading TTT Model Weights ==="
# Ensure the script is executable
chmod +x download_model.sh
./download_model.sh

echo "=== 3. Running Inference Verification (Wikitext + Public Llama 2) ==="
# We use the public tokenizer to avoid login prompts
python -m ttt.train \
    --mesh_dim='1,1,1' \
    --dtype="bf16" \
    --eval_mode=True \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-103-v1" \
    --tokenizer_name="nousresearch/Llama-2-7b-hf" \
    --dataset_path="./data_cache" \
    --load_model_config="experiments/1b-TTT/metadata.pkl" \
    --load_part="trainstate" \
    --exp_dir="experiments" \
    --exp_name="1b-TTT" \
    --resume_exp_name="1b-TTT" \
    --seq_length=2048 \
    --global_batch_size=1

echo "=== SUCCESS! Environment is ready. ==="