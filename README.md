# Mechanistic Interpretability of Test-Time Training (TTT) Layers

This repository contains code for **"A Functional Decomposition of Dynamic Plasticity"** — a mechanistic interpretability study of Test-Time Training (TTT) layers in language models.

Based on: [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)

## Abstract

Test-Time Training (TTT) represents a paradigm shift from static inference to dynamic adaptation. Unlike standard Transformers that rely on attention-based activation dynamics, TTT layers treat the hidden state as mutable weights updated via gradient descent during inference. This enables efficient long-context modeling with $O(1)$ memory cost while maintaining higher capacity than fixed-dimensional state vectors.

However, current evaluations treat TTT as a "black box," reporting aggregate improvements without explaining *what* is being learned during inference. This work introduces **Functional Plasticity Bucketing**, a methodology that decomposes the aggregate loss into specific token categories to understand the mechanistic drivers of TTT plasticity.

We investigate three core research questions:
- **RQ1 (The Trigger):** What linguistic and temporal features drive the gradient descent process?
- **RQ2 (The Location):** Which layers are responsible for long-context memory?
- **RQ3 (The Efficiency):** Does TTT exhibit selective plasticity, decoupling surprisal from update magnitude?

For detailed results and visualizations, see [`docs/index.html`](docs/index.html).

## Methodology: Functional Plasticity Bucketing

We define **Plasticity** as the Frobenius norm of the weight update, denoted as $||\Delta \theta||_F$, at a given timestep. To understand what drives these updates, we classify every token into three functional buckets:

| Bucket | Category | Definition | Hypothesis |
|--------|----------|------------|------------|
| **A** | **Syntax** | High-frequency stop words ("the", "of") | Zero / Negligible plasticity. Base weights should already handle these stationary patterns. |
| **B** | **Retrieval** | Content words seen previously in context | High plasticity (Binding). Re-occurrence triggers updates to "bind" current context to stored representation. |
| **C** | **Novelty** | Content words appearing for the first time | Highest plasticity (Surprisal). High information content drives the TTT layer to learn new entities. |

This bucketing approach allows us to test the **Plasticity Hypothesis**: that the magnitude of dynamic updates is a direct proxy for the information content of the token, grounded in Information Theory.

## Setup

This codebase is implemented in [JAX](https://jax.readthedocs.io/en/latest/index.html) and has been tested on both GPUs and Cloud TPU VMs with Python 3.11.

For a PyTorch model definition, please refer to [this link](https://github.com/test-time-training/ttt-lm-pytorch). For inference kernels, or to replicate speed benchmarks from our paper, please view our [kernel implementations](https://github.com/test-time-training/ttt-lm-kernels).

### Environment Installation

To setup and run our code on a (local) GPU machine, we highly recommend using [Anaconda](https://anaconda.com/download) when installing python dependencies. Install GPU requirements using:
```
cd TTT-Code/requirements
pip install -r gpu_requirements.txt
```

For TPU, please refer to [this link](https://cloud.google.com/tpu/docs/quick-starts) for guidance on creating cloud TPU VMs. Then, run:
```
cd TTT-Code/requirements
pip install -r tpu_requirements.txt
```

### WandB Login

We use WandB for logging training metrics and TTT statistics. After installing the requirements, login to WandB using:
```
wandb login
```

### Dataset Download

**Model:** All experiments in this study use a **Llama 1.3B model trained on Books3**.

**Analysis Datasets:** For the mechanistic interpretability experiments, we use:
- **PG-19** (Project Gutenberg): Natural long-context text for studying real-world narrative dependencies
- **Synthetic "Needle-in-a-Haystack"** tasks: Controlled retrieval experiments for precise mechanistic analysis

The analysis scripts (`run_inference.py`) support both dataset types and can stream PG-19 directly or generate synthetic data on-the-fly.

**Training Datasets:** For training TTT models, Llama-2 tokenized datasets are available for download from Google Cloud Buckets:

```
gsutil -m cp -r gs://llama2-pile/* llama-2-pile/
gsutil -m cp -r gs://llama2-books/* llama-2-books3/
```

Once downloaded, set the `dataset_path` flag in `train.py` to the directory containing the `tokenizer_name-meta-llama` folder. This will allow the dataloader to find the correct path.

Alternatively, to tokenize datasets yourself, refer to [dataset preparation](TTT-Code/ttt/dataloader/README.md).

## Replicating Experiments

### Training Experiments

We provide scripts corresponding to each training experiment in the `TTT-Code/scripts` folder. After specifying the experiment name and directory, select the desired context length and divide by 0.5 million to calculate the appropriate batch size.

Depending on the model size, you may need to modify the `mesh_dim` to introduce model sharding. See the [model docs](TTT-Code/ttt/README.md) for additional information on the training configuration.

### Mechanistic Analysis Experiments

#### Prerequisites

Before running the analysis experiments, ensure you have:
1. **Trained Model Checkpoint**: A checkpoint from a Llama 1.3B TTT model trained on Books3
2. **Model Config**: The model configuration pickle file (typically saved as `metadata.pkl` in the experiment directory)
3. **Tokenizer**: The script uses `nousresearch/Llama-2-7b-hf` tokenizer by default (downloads automatically)

#### Running Plasticity Analysis (RQ1 & RQ3)

The main analysis script [`TTT-Code/scripts/analysis/run_inference.py`](TTT-Code/scripts/analysis/run_inference.py) generates plasticity statistics and bucketed metrics. Navigate to the `TTT-Code` directory before running:

**PG-19 Experiment (Natural Long-Context):**
```bash
cd TTT-Code
python3 -m scripts.analysis.run_inference \
    --load_checkpoint="experiments/1b-TTT/streaming_train_state_1000" \
    --load_model_config="experiments/1b-TTT/metadata.pkl" \
    --dataset_type="pg19" \
    --tokenizer_name="nousresearch/Llama-2-7b-hf" \
    --seq_length=32768 \
    --global_batch_size=1 \
    --max_steps=10 \
    --exp_dir="analysis_outputs" \
    --exp_name="pg19_baseline" \
    --output_filename="results.pkl" \
    --mesh_dim="-1,1,1" \
    --dtype="bf16"
```

**Note:** The `cd TTT-Code` command is required because the script imports from the `ttt` package. All paths in the command are relative to the `TTT-Code` directory.

**Synthetic "Needle-in-a-Haystack" Experiment:**
```bash
cd TTT-Code
python3 -m scripts.analysis.run_inference \
    --load_checkpoint="experiments/1b-TTT/streaming_train_state_1000" \
    --load_model_config="experiments/1b-TTT/metadata.pkl" \
    --dataset_type="synthetic" \
    --tokenizer_name="nousresearch/Llama-2-7b-hf" \
    --seq_length=32768 \
    --global_batch_size=1 \
    --max_steps=10 \
    --exp_dir="analysis_outputs" \
    --exp_name="verify_synthetic" \
    --output_filename="results.pkl" \
    --mesh_dim="-1,1,1" \
    --dtype="bf16"
```

**Key Parameters:**
- `--load_checkpoint`: Path to the trained model checkpoint directory
- `--load_model_config`: Path to the model configuration pickle file
- `--dataset_type`: Either `"pg19"` (streams from HuggingFace) or `"synthetic"` (generates on-the-fly)
- `--seq_length`: Sequence length (use 32768 for long-context experiments)
- `--max_steps`: Number of batches to process (adjust based on compute budget)
- `--exp_dir` and `--exp_name`: Output directory structure
- `--mesh_dim`: JAX mesh configuration (use `"-1,1,1"` for single device)

**Output Format:**
The script generates a pickle file containing:
- Per-token gradient norms for each layer (`raw_grads`)
- Bucketed metrics: average plasticity for Syntax, Retrieval, and Novelty tokens per layer
- Overall loss per step
- Input tokens for reference

**Verifying Results:**

Use [`TTT-Code/scripts/analysis/check_results.py`](TTT-Code/scripts/analysis/check_results.py) to verify analysis outputs:

```bash
# From repository root:
python3 TTT-Code/scripts/analysis/check_results.py TTT-Code/analysis_outputs/pg19_baseline/results.pkl

# Or from TTT-Code directory:
cd TTT-Code
python3 scripts/analysis/check_results.py analysis_outputs/pg19_baseline/results.pkl
```

This script performs three sanity checks:
1. **Shape Verification**: Confirms gradient norms match sequence length (per-token resolution)
2. **Bucketing Sanity**: Verifies that token categories have distinct plasticity values
3. **Temporal Decay Check**: Observes whether plasticity decreases over time (burn-in phase)

Expected output should show:
- ✅ SUCCESS messages for shape and bucketing verification
- 📉 Observation of plasticity decay over time (for long sequences)

#### Layer Freezing Experiments (RQ2)

To investigate which layers are responsible for long-context memory, we freeze specific layer ranges by setting `frozen_layers` in the model config. This disables TTT updates (sets $\eta=0$) for the specified layers.

**Prerequisites:**
- Trained TTT model checkpoint (same as above)
- Model config file
- Dataset path configured (wikitext or custom)

**Running Freezing Experiments:**

Scripts are provided in [`TTT-Code/scripts/ttt_linear/freezing_experiments/`](TTT-Code/scripts/ttt_linear/freezing_experiments/). Each script freezes different layer patterns:

- `1.3b_wikitext_middle_light.sh`: Freezes layers [0,1,22,23] (middle layers 2-21 active)
- `1.3b_wikitext_middle_strict.sh`: More restrictive middle-layer configuration
- `1.3b_cortex.sh`: Freezes early layers [0,1,2,3,4] (cortex-like pattern)
- `1.3b_retina.sh`: Freezes late layers (retina-like pattern)
- `1.3b_drift_uniform.sh`: Uniform layer freezing pattern
- `1.3b_drift_retina.sh`: Retina pattern with drift
- `1.3b_wikitext.sh`: Baseline wikitext configuration (no freezing)

**Example Usage:**
```bash
cd TTT-Code/scripts/ttt_linear/freezing_experiments
bash 1.3b_wikitext_middle_light.sh
```

**Custom Freezing Pattern:**

To create a custom freezing experiment, modify the `--update_model_config` flag:

```bash
python3 -m ttt.train \
    --load_model_config="experiments/1b-TTT/metadata.pkl" \
    --load_part="trainstate_params" \
    --resume_exp_name="1b-TTT" \
    --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0, frozen_layers=[0, 1, 2, 3])" \
    --dataset_path="./data" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-103-v1" \
    --seq_length=2048 \
    --global_batch_size=8 \
    --eval_mode=True \
    --total_steps=20 \
    --exp_dir="./experiments" \
    --exp_name="custom_freezing"
```

**Interpreting Results:**

Compare the loss from freezing experiments to the baseline (no freezing):
- **Higher loss** when freezing critical layers indicates those layers are essential for the task
- **PG-19**: Middle layers (8-15) should show highest degradation when frozen
- **Synthetic**: Late layers (16-23) should show highest degradation when frozen

These experiments test the hypothesis that middle layers handle long-range semantic binding while upper layers handle precise retrieval.

## Research Findings Summary

### RQ1: The Trigger (Linguistic and Temporal Drivers)

**Finding:** Novelty tokens exhibit the highest plasticity (approximately 28% higher than Syntax tokens), confirming that TTT plasticity is event-driven and primarily concerned with encoding new information.

**Temporal Pattern:** The TTT layer exhibits a distinct "Burn-in Phase" (first ~2,000 tokens) where gradient norms are 2x higher than the steady state, suggesting rapid adaptation to document style followed by selective maintenance updates.

### RQ2: The Location (Layer Localization)

**Finding:** Memory localization is **task-dependent**:
- **Natural Narrative (PG-19):** Relies on **Middle Layers (8-15)** for maintaining coherent representations of characters, settings, and plotlines.
- **Algorithmic Retrieval (Synthetic):** Relies on **Late Layers (16-23)** for precise extraction of arbitrary passkeys.

This reveals a division of labor: middle layers handle semantic binding, while upper layers handle task-specific reasoning and output formatting.

### RQ3: The Efficiency (Selective Plasticity)

**Finding:** Near-zero correlation ($r \approx 0.03-0.05$) between model surprisal (Cross Entropy loss) and update magnitude (Gradient Norm). This contradicts the naive view of TTT as a simple error-correction loop.

**Implication:** The outer-loop optimization has learned to distinguish between "useful surprise" (novel entities worth encoding) and "noise" (irreducible uncertainty). This **meta-learned filter** enables selective plasticity without heuristic gating mechanisms.

For detailed visualizations and extended discussion, see [`docs/index.html`](docs/index.html).

## Credits

* This mechanistic interpretability study is based on the original TTT work: [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) by Sun et al. (2024).
* This codebase is based on [EasyLM](https://github.com/young-geng/EasyLM).
* Our dataloader is based on [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main/training).
