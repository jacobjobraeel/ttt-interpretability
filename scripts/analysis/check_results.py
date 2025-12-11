import pickle
import numpy as np
import sys
import os

def check_results(path):
    print(f"Loading results from {path}...")
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    with open(path, "rb") as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} steps.")
    
    if len(results) == 0:
        print("Error: Results list is empty.")
        return

    # 1. Shape Verification
    first_step = results[0]
    layer_idx = 0 # Check layer 0
    
    # Handle cases where raw_grads might not exist or be empty
    if "raw_grads" not in first_step or layer_idx not in first_step["raw_grads"]:
        print("Error: 'raw_grads' not found in results.")
        return

    raw_grad = first_step["raw_grads"][layer_idx]

    # Flatten raw_grad if needed: (B, 128, 16) -> (B*128*16) or just (B, 2048)
    # We want to check per-sequence length.
    # Assuming batch size 1 for verification script logic simplicity, or just flatten everything.
    flat_grad = raw_grad.flatten()
    
    print("\n=== 1. Shape Verification ===")
    print(f"Layer {layer_idx} raw_grads shape: {raw_grad.shape}")
    print(f"Flattened shape: {flat_grad.shape}")
    
    # input_tokens is (Batch, Seq_Len)
    expected_len = first_step['input_tokens'].size 
    print(f"Expected sequence length (total tokens): {expected_len}")
    
    if flat_grad.size == expected_len:
        print("✅ SUCCESS: Gradient norm size matches sequence length (Per-token resolution confirmed).")
    else:
        print("❌ FAIL: Shape mismatch!")

    # 2. Bucketing Sanity
    print("\n=== 2. Bucketing Sanity (Layer 0) ===")
    
    avg_syn = np.mean([step["layers"][layer_idx]["grad_syntax"] for step in results])
    avg_ret = np.mean([step["layers"][layer_idx]["grad_retrieval"] for step in results])
    avg_nov = np.mean([step["layers"][layer_idx]["grad_novelty"] for step in results])
    
    print(f"Avg Syntax Plasticity:    {avg_syn:.6f}")
    print(f"Avg Retrieval Plasticity: {avg_ret:.6f}")
    print(f"Avg Novelty Plasticity:   {avg_nov:.6f}")
    
    if avg_syn != avg_ret:
        print("✅ SUCCESS: Categories have distinct values (Masking is working).")
    else:
        print("❌ FAIL: All categories are identical! (Likely averaged over batch or seq).")

    # 3. Decay Check
    print("\n=== 3. Temporal Decay Check ===")
    # Stack all raw grads: (n_steps, B, 128, 16)
    # We want mean over steps and batch, preserving time.
    # Flatten to (n_steps * B, Seq_Len)
    all_grads = []
    for step in results:
        g = step["raw_grads"][layer_idx] # (B, 128, 16)
        g_flat = g.reshape(g.shape[0], -1) # (B, 2048)
        all_grads.append(g_flat)
        
    all_grads = np.vstack(all_grads) # (Total_Samples, 2048)
    mean_trace = np.mean(all_grads, axis=0) # (2048,)
    
    start_mean = mean_trace[:100].mean()
    end_mean = mean_trace[-100:].mean()
    
    print(f"Mean Plasticity (First 100 tokens): {start_mean:.6f}")
    print(f"Mean Plasticity (Last 100 tokens):  {end_mean:.6f}")
    
    if start_mean > end_mean:
        print("📉 Observation: Plasticity decreases over time (Decay).")
    else:
        print("📈 Observation: Plasticity increases or is stable.")

if __name__ == "__main__":
    path = "analysis_outputs/verify_synthetic/results.pkl"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    check_results(path)
