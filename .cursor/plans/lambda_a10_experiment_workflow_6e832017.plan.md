---
name: Lambda A10 Experiment Workflow
overview: Complete workflow for setting up a Lambda A10 GPU instance and running all 5 TTT experiments (wikitext baseline, cortex, retina, drift-uniform, drift-retina).
todos:
  - id: ssh-connect
    content: SSH into Lambda A10 instance
    status: pending
  - id: clone-repo
    content: Clone the ttt-lm-jax repository
    status: pending
  - id: run-setup
    content: Run setup_and_run.sh to install deps and download model
    status: pending
  - id: run-wikitext
    content: Run 1.3b_wikitext.sh baseline experiment
    status: pending
  - id: run-cortex
    content: Run 1.3b_cortex.sh (freeze early layers)
    status: pending
  - id: run-retina
    content: Run 1.3b_retina.sh (freeze late layers)
    status: pending
  - id: run-drift-uniform
    content: Run 1.3b_drift_uniform.sh (repetitive data)
    status: pending
  - id: run-drift-retina
    content: Run 1.3b_drift_retina.sh (repetitive + freeze late)
    status: pending
---

# Lambda A10 Experiment Workflow

## Step 1: Start and Connect to Lambda Instance

```bash
# SSH into your Lambda A10 instance
ssh ubuntu@<your-lambda-ip>
```



## Step 2: Clone the Repository

```bash
git clone https://github.com/<your-repo>/ttt-lm-jax.git
cd ttt-lm-jax
```



## Step 3: Run Setup Script (One Command)

The `setup_and_run.sh` script handles everything:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

This script will:

1. Create a Python virtual environment (`venv`)
2. Install GPU dependencies (JAX with CUDA 12 + cuDNN 9.1)
3. Download TTT-1.3B model weights (~7.7GB) to `experiments/1b-TTT/`
4. Run a verification inference to confirm setup works

## Step 4: Activate Environment (for subsequent sessions)

```bash
source venv/bin/activate
```



## Step 5: Run Experiments

Run experiments sequentially (each takes ~5-15 minutes):

```bash
# 1. Wikitext Baseline (no frozen layers)
bash scripts/ttt_linear/1.3b_wikitext.sh

# 2. Cortex Experiment (freeze layers 0-4)
bash scripts/ttt_linear/1.3b_cortex.sh

# 3. Retina Experiment (freeze layers 20-23)
bash scripts/ttt_linear/1.3b_retina.sh

# 4. Drift Uniform (repetitive data, no freeze)
bash scripts/ttt_linear/1.3b_drift_uniform.sh

# 5. Drift Retina (repetitive data, freeze layers 20-23)
bash scripts/ttt_linear/1.3b_drift_retina.sh
```



## Step 6: Check Results

Results are saved in `experiments/<exp_name>/`:

- Checkpoints at milestone steps
- Logs viewable on WandB (if logged in)
```bash
ls experiments/
# Expected: 1b-TTT, 1.3b-wikitext-baseline, 1.3b-cortex, 1.3b-retina, etc.
```




## Quick Reference: Experiment Summary

| Script | Experiment Name | Frozen Layers | Steps | Data Type ||--------|----------------|---------------|-------|-----------|| `1.3b_wikitext.sh` | 1.3b-wikitext-baseline | None | 20 | Wikitext || `1.3b_cortex.sh` | 1.3b-cortex | 0-4 | 20 | Wikitext || `1.3b_retina.sh` | 1.3b-retina | 20-23 | 20 | Wikitext || `1.3b_drift_uniform.sh` | 1.3b-drift-uniform | None | 50 | Repetitive || `1.3b_drift_retina.sh` | 1.3b-drift-retina | 20-23 | 50 | Repetitive |