# Handover Prompt: TTT Plasticity Analysis Project

**Project Goal:**
We are conducting a scientific analysis of **Test-Time Training (TTT)** layers (specifically `ttt_linear` in a 1.3B model). We are moving beyond simple perplexity metrics to understand the *mechanics* of plasticity: **Where** (which layers), **When** (at what tokens), and **Why** (syntax vs. memory) the model updates its weights during inference.

**Current State (Codebase):**
We have successfully implemented the "Phase 1" analysis infrastructure.
1.  **Script:** `scripts/analysis/run_inference.py` has been completely rewritten.
    *   **Bucketing Logic:** It now classifies every token as `Syntax` (Stop word), `Retrieval` (Seen before), or `Novelty` (New).
    *   **Metrics:** It logs `grad_norm` and `loss` separately for each bucket.
2.  **Datasets:** We removed Wikitext (too short) and added:
    *   **Synthetic:** "Needle in a Haystack" generator (hardcoded in `run_inference.py`) to validate retrieval mechanics.
    *   **PG-19:** Streaming support for long-context books (via HuggingFace) to test natural OOD performance.
3.  **Experiments:** We created two runner scripts in `scripts/analysis/experiments/`:
    *   `verify_synthetic.sh`: Quick sanity check (10 steps).
    *   `run_pg19_baseline.sh`: Main data generator for heatmaps (50 steps).

**Technical Context (JAX/TTT):**
*   **Model:** 1.3B parameter TTT-Linear model (JAX/Flax).
*   **Key Concept:** `model.apply` is **NOT static**. It runs an inner loop gradient descent on the hidden state $W_t$ for every sequence. We are capturing these inner gradients (`grad_norm_t`) to visualize plasticity.
*   **Issues Fixed:** We just fixed a bug where the shell scripts invoked `python -m scripts...` but `scripts/` lacked an `__init__.py`. We agreed to add `__init__.py` files to fix this.

**Immediate Next Steps (To-Do for New Session):**

1.  **Fix Module Imports:**
    *   Add empty `__init__.py` files to `scripts/` and `scripts/analysis/` so `python3 -m scripts.analysis.run_inference` works correctly.

2.  **Run Verification:**
    *   Run `./scripts/analysis/experiments/verify_synthetic.sh` on the GPU instance.
    *   *Success Criteria:* Ensure the output `.pkl` file contains `grad_retrieval` values that are non-zero and distinct from `grad_syntax`.

3.  **Run PG-19 Baseline:**
    *   Edit `scripts/analysis/experiments/run_pg19_baseline.sh` to point to the actual checkpoint path (`load_checkpoint=...`).
    *   Run the script to generate the data for the "Plasticity Heatmap".

4.  **Visualization:**
    *   Create a new script `scripts/analysis/plot_results.py` to load the `.pkl` files and generate the Heatmap (Layer x Time) and Bar Charts (Bucket performance).

**Key Research Hypotheses to Verify:**
*   **RQ1:** TTT updates (plasticity) should be sparse and concentrated on "Novelty" and "Retrieval" tokens, not "Syntax".
*   **RQ2:** Middle/Deep layers should show higher plasticity for long-range retrieval than early layers.
