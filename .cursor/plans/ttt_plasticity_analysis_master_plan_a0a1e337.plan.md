---
name: TTT Plasticity Analysis Master Plan
overview: Implement a master plan to guide the creation of analysis tools and visualizations that investigate TTT model plasticity, addressing the user's questions about logging, architecture roles, and dataset suitability.
todos:
  - id: subplan-a
    content: "Create Sub-Plan A: Build `scripts/analysis/run_inference.py` infrastructure."
    status: pending
  - id: subplan-b
    content: "Create Sub-Plan B: Implement Synthetic Data generator in the inference script."
    status: pending
  - id: subplan-c
    content: "Create Sub-Plan C: Build `scripts/analysis/plot_results.py` for the 4 figures."
    status: pending
---

# TTT Plasticity Analysis Master Plan

This master plan outlines the strategy to create a dedicated analysis pipeline for investigating "Why" TTT models behave as they do. It breaks down the work into sub-plans to address specific concerns: decoupling from the training codebase, simplifying logging (removing WandB dependency), and validating findings with appropriate datasets.

## Phase 1: Decoupling & Inference Infrastructure

**Goal:** Create a standalone inference script that reuses model definitions but avoids `train.py` bloat and WandB complexity.

1.  **Create `scripts/analysis/run_inference.py`**:

    -   **Objective:** Load a checkpoint and run a forward pass with `output_ttt_stats=True`.
    -   **Implementation:**
        -   Import `CausalLM` and `ModelConfig` from `ttt.models.model`.
        -   Import `StreamingCheckpointer` from `ttt.infra.checkpoint`.
        -   Implement a minimal `main` loop that iterates over a dataset.
        -   **Crucial Change:** Instead of `wandb.log`, accumulate `grad_norm_t`, `loss`, `token_ids`, and `layer_indices` into a local dictionary.
        -   Save results to `results/{exp_name}/analysis_stats.pkl`.

## Phase 2: Visualization & Storytelling (The 4 Figures)

**Goal:** Generate the 4 key figures requested to explain the model's internal mechanics ("Bertology" for TTT).

1.  **Create `scripts/analysis/plot_results.py`**:

    -   **Objective:** Load `analysis_stats.pkl` and generate static `.png` figures.
    -   **Figure 1: The Plasticity Heatmap**
        -   *Data:* `grad_norm_t` matrix (Layer x Time).
        -   *Story:* "Updates are not uniform; they concentrate in [specific layers] at [specific times]."
    -   **Figure 2: Surprise vs. Learning**
        -   *Data:* Scatter plot of $Loss_{t-1}$ vs $GradNorm_t$.
        -   *Story:* "The model 'pays attention' (updates weights) primarily when it encounters high-surprisal tokens."
    -   **Figure 3: Syntax vs. Semantics**
        -   *Data:* Bar chart comparing average $GradNorm$ for Stop Words vs. Content Words (requires simple POS tagging or frequency bucket heuristic).
        -   *Story:* "TTT layers ignore 'the' and 'and', focusing updates on rare entities."
    -   **Figure 4: Layer Role Ablation (Cumulative Adaptation)**
        -   *Data:* Line plot of Perplexity over Token Index for different freezing configurations (using data from your existing bash scripts, or re-running them via this new script).