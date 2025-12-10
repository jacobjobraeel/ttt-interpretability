#!/bin/bash

DATA_PATH="./data"
DATA_NAME="wikitext"
DATA_CONFIG="wikitext-103-v1"

# Product should equal 0.5 million
SEQ_LEN=2048
BS=8

# Experiment details
EXP_NAME="1.3b-wikitext-middle-strict"
EXP_DIR="./experiments"

mkdir -p ${EXP_DIR}/${EXP_NAME}
chmod -R 777 ${EXP_DIR}/${EXP_NAME} 2>/dev/null || true

# Use platform allocator to avoid memory fragmentation and preallocation issues
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python3 -m ttt.train \
        --mesh_dim='!-1,1,1' \
        --dtype='fp32' \
        --eval_mode=True \
        --total_steps=20 \
        --save_checkpoint_freq=0 \
        --save_milestone_freq=1 \
        --load_model_config="pickle::experiments/1b-TTT/metadata.pkl" \
        --load_part="trainstate_params" \
        --resume_exp_name="1b-TTT" \
        --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0, frozen_layers=[0, 1, 2, 3, 20, 21, 22, 23])" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --dataset_config_name=${DATA_CONFIG} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=1e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=5 \
        --optimizer.adamw_optimizer.lr_decay_steps=20 \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}

