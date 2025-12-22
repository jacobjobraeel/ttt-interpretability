import mlxu
import os
import os.path as osp
import time
import pickle
import numpy as np
import uuid
from tqdm import tqdm
from copy import deepcopy

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.experimental.multihost_utils import process_allgather
import flax
import optax
from flax.training.train_state import TrainState

from transformers import AutoTokenizer
from datasets import load_dataset

from ttt.dataloader.language_modeling_hf import LMDataModule

# 1. Define the words (Top 100 English words)
# Source: Wikipedia / Oxford English Corpus standard lists
TOP_100_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", 
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", 
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", 
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", 
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", 
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", 
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", 
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    ".", ",", "?", "!", ":", ";", "-", "\n" # Add punctuation manually
]

def get_syntax_ids(tokenizer):
    syntax_ids = set()
    for word in TOP_100_WORDS:
        # Llama 2 treats "word" and " word" (space prefix) differently
        # We want to catch both.
        
        # 1. Strict encoding (no special tokens added)
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1: 
            syntax_ids.add(ids[0])
            
        # 2. Leading space version (" the")
        ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids_space) == 1:
            syntax_ids.add(ids_space[0])
            
        # 3. Capitalized version ("The") - Optional but good for start of sentence
        ids_cap = tokenizer.encode(word.capitalize(), add_special_tokens=False)
        if len(ids_cap) == 1:
            syntax_ids.add(ids_cap[0])
            
    return syntax_ids

class SyntheticDataModule(LMDataModule):
    def __init__(self, max_steps, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps

    def prepare_data(self):
        pass

    def process_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        master_print("Generating Synthetic 'Needle in Haystack' data...")
        
        all_token_ids = []
        # Generate enough data for max_steps * global_batch_size
        # We estimate needed sequences. A bit more to be safe.
        num_sequences = self.max_steps * self.batch_size + 4 # Arbitrary buffer
        
        for _ in range(num_sequences):
            # 1. Generate Needle
            key = str(uuid.uuid4())[:8] # Shorten for readability
            val = str(uuid.uuid4())[:8]
            
            prefix_text = f"Define Key={key} Value={val}. "
            query_text = f" Query Key={key} Value="
            
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
            val_tokens = tokenizer.encode(val, add_special_tokens=False)
            
            # 2. Calculate Filler
            # total = seq_len
            # filler = total - prefix - query - val
            current_len = len(prefix_tokens) + len(query_tokens) + len(val_tokens)
            filler_len = self.max_length - current_len
            
            if filler_len < 0:
                # Should not happen with large seq_len, but safe check
                filler_len = 0
            
            # 3. random noise filler
            # range of vocab (approx 32000 for llama)
            filler_tokens = np.random.randint(100, 32000, size=filler_len).tolist()
            
            # Construct sequence: Prefix + Filler + Query + Value
            # Actually we want the retrieval to happen at the END.
            # So: Prefix ... Filler ... Query -> Value
            
            seq = prefix_tokens + filler_tokens + query_tokens + val_tokens
            
            # Truncate to exact length if slightly over due to math
            seq = seq[:self.max_length]
            
            all_token_ids.extend(seq)
            
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
        full_ids = np.array(all_token_ids, dtype=dtype)
        
        return {
            "train": np.array([], dtype=dtype),
            "validation": full_ids,
            "test": np.array([], dtype=dtype),
        }, tokenizer

class PG19DataModule(LMDataModule):
    def prepare_data(self):
        # Override to prevent base class from downloading full dataset
        pass

    def process_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        master_print(f"Streaming PG-19 (first 5 books)...")
        
        dataset = load_dataset("emozilla/pg19", split="test", streaming=True)
        subset = list(dataset.take(5))
        
        all_token_ids = []
        for i, book in enumerate(subset):
            text = book['text']
            # Take a slice of text to avoid huge tokenization? 
            # PG19 books are long. Let's take first 200k chars to be safe/fast
            # and ensure we fill 32k tokens.
            # Avg char/token ~ 4. 32k tokens ~ 128k chars. 
            text_slice = text[:200000] 
            
            tokens = tokenizer.encode(text_slice, add_special_tokens=False)
            
            # We want strictly sequences of seq_len. 
            # LMDataset handles slicing, but let's ensure we have enough.
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            all_token_ids.extend(tokens)
            
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
        full_ids = np.array(all_token_ids, dtype=dtype)
        
        return {
            "train": np.array([], dtype=dtype),
            "validation": full_ids,
            "test": np.array([], dtype=dtype),
        }, tokenizer
from ttt.dataloader.quick_loader import QuickTestDataModule
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.models.model import ModelConfig, CausalLM
from ttt.infra.jax_utils import (
    JaxRNG,
    JaxDistributedConfig,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    make_shard_and_gather_fns,
    with_sharding_constraint,
    master_print,
    set_random_seed,
    get_float_dtype_by_name,
)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=0,
    mesh_dim="-1,1,1", # Default to no sharding for simplicity, or adjusted by user
    dtype="bf16",
    load_checkpoint="", # Path to checkpoint directory (e.g. .../streaming_train_state_1000)
    dataset_path="",
    dataset_type="pg19", # "pg19" or "synthetic"
    dataset_name="the_pile",
    dataset_config_name="wikitext-103-v1",
    tokenizer_name="nousresearch/Llama-2-7b-hf",
    seq_length=2048,
    global_batch_size=1, # Default to 1 for detailed analysis
    loader_workers=4,
    exp_dir="results",
    exp_name="analysis_default",
    jax_distributed=JaxDistributedConfig.get_default_config(),
    load_model_config="",
    update_model_config="",
    max_steps=10, # Limit number of steps to analyze
    dataset_split="validation", # or "test"
    output_filename="analysis_stats.pkl",
)

def make_inference_step_fn(model, model_config):
    def inference_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        
        outputs = model.apply(
            train_state.params,
            batch["input_tokens"],
            deterministic=True,
            output_ttt_stats=True, # We always want stats for analysis
            rngs=rng_generator(model_config.rng_keys()),
        )
        
        logits = outputs.logits
        ttt_stats = outputs.ttt_stats
        
        # Calculate loss per token (unreduced) for detailed analysis if needed, 
        # but cross_entropy_loss_and_accuracy usually returns mean scalar.
        # We might want the per-token loss.
        # Let's use the standard loss for now to match training.
        loss, _ = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])
        
        return rng_generator(), loss, ttt_stats, logits

    return inference_step

def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    
    set_random_seed(FLAGS.seed)
    master_process = jax.process_index() == 0
    master_print(f"Running inference analysis on {FLAGS.dataset_name}...")

    # Data Module
    if FLAGS.dataset_type == "synthetic":
        data_module = SyntheticDataModule(
            max_steps=FLAGS.max_steps,
            dataset_name="synthetic",
            dataset_config_name=None,
            tokenizer_name=FLAGS.tokenizer_name,
            cache_dir=None,
            max_length=FLAGS.seq_length,
            add_eos=True,
            batch_size=FLAGS.global_batch_size,
            batch_size_eval=FLAGS.global_batch_size,
            loader_workers=FLAGS.loader_workers,
            shuffle=False,
            fault_tolerant=False,
            drop_last=True,
        )
    elif FLAGS.dataset_type == "pg19":
        data_module = PG19DataModule(
            dataset_name="deepmind/pg19",
            dataset_config_name=None,
            tokenizer_name=FLAGS.tokenizer_name,
            cache_dir=None,
            max_length=FLAGS.seq_length,
            add_eos=True,
            batch_size=FLAGS.global_batch_size,
            batch_size_eval=FLAGS.global_batch_size,
            loader_workers=FLAGS.loader_workers,
            shuffle=False,
            fault_tolerant=False,
            drop_last=True,
        )
    elif FLAGS.dataset_path == "quick_test":
        data_module = QuickTestDataModule(
            dataset_name=FLAGS.dataset_name,
            dataset_config_name=FLAGS.dataset_config_name,
            tokenizer_name=FLAGS.tokenizer_name,
            cache_dir=None,
            max_length=FLAGS.seq_length,
            add_eos=True,
            batch_size=FLAGS.global_batch_size,
            batch_size_eval=FLAGS.global_batch_size,
            loader_workers=FLAGS.loader_workers,
            shuffle=False,
            fault_tolerant=False,
            drop_last=True,
        )
    else:
        data_module = LMDataModule(
            dataset_name=FLAGS.dataset_name,
            dataset_config_name=FLAGS.dataset_config_name,
            tokenizer_name=FLAGS.tokenizer_name,
            cache_dir=FLAGS.dataset_path,
            max_length=FLAGS.seq_length,
            add_eos=True,
            batch_size=FLAGS.global_batch_size,
            batch_size_eval=FLAGS.global_batch_size,
            loader_workers=FLAGS.loader_workers,
            shuffle=False,
            fault_tolerant=False,
            drop_last=True,
        )

    data_module.prepare_data()
    data_module.setup()
    
    if FLAGS.dataset_split == "validation":
        loader = data_module.val_dataloader()
    elif FLAGS.dataset_split == "test":
        loader = data_module.test_dataloader()
    else:
        # Fallback to train if needed, though unusual for analysis
        loader = data_module.train_dataloader()

    # Model Config
    if FLAGS.load_model_config != "":
        model_config = ModelConfig.load_config(FLAGS.load_model_config)
    else:
        raise RuntimeError("load_model_config must be specified")

    if FLAGS.update_model_config:
        update_dic = eval(FLAGS.update_model_config)
        for key, value in update_dic.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                raise KeyError(f"Update key {key} not in model_config")
    
    model_config.vocab_size = data_module.vocab_size
    model_config.max_sequence_length = FLAGS.seq_length

    # Model Initialization
    model_dtype = get_float_dtype_by_name(FLAGS.dtype)
    model = CausalLM(model_config, dtype=model_dtype, param_dtype=model_dtype)

    # Sharding
    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, FLAGS.seq_length), dtype=jnp.int32),
            rngs=rng_generator(model_config.rng_keys()),
        )
        # Manually create TrainState with no optimizer
        return TrainState(step=0, params=params, tx=None, opt_state=None, apply_fn=None)

    # Checkpointer
    checkpointer = StreamingCheckpointer(StreamingCheckpointer.get_default_config(), FLAGS.load_checkpoint, enable=master_process)

    # Compile sharded functions
    inference_step = make_inference_step_fn(model, model_config)
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(model_config.get_partition_rules(), train_state_shapes)
    
    shard_fns, _ = make_shard_and_gather_fns(train_state_partition, train_state_shapes)

    sharded_init_fn = pjit(init_fn, in_shardings=PS(), out_shardings=train_state_partition)
    
    # We need a way to load params into the sharded state
    # We can use StreamingCheckpointer's load_trainstate_checkpoint
    # It returns a TrainState. 
    
    mesh = model_config.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        sharded_rng = next_rng()
        
        # Load Checkpoint
        master_print(f"Loading checkpoint from {FLAGS.load_checkpoint}...")
        
        # We need to construct the target for loading. 
        # sharded_init_fn gives us a structure with abstract values if we just eval_shape, 
        # but here we need the actual structure to load into.
        # Actually StreamingCheckpointer.load_trainstate_checkpoint takes shapes/shard_fns.
        
        # Note: StreamingCheckpointer expects a path like "trainstate::/path/to/ckpt"
        # We'll assume FLAGS.load_checkpoint is the directory, and we prepend the type.
        # If the user passed the full string "trainstate::...", we handle it.
        load_path = FLAGS.load_checkpoint
        if "::" not in load_path:
            load_path = f"trainstate_params::{load_path}"
            
        train_state, restored_params = checkpointer.load_trainstate_checkpoint(
            load_path, train_state_shapes, shard_fns
        )
        
        if train_state is None and restored_params is not None:
             # Create TrainState from params
             # We need a pjit function to create TrainState from params
             def create_train_state(params):
                 # Manually create TrainState with no optimizer
                 return TrainState(step=0, params=params, tx=None, opt_state=None, apply_fn=None)
                 
             sharded_create = pjit(
                 create_train_state, 
                 in_shardings=(train_state_partition.params,), 
                 out_shardings=train_state_partition
             )
             train_state = sharded_create(flax.core.unfreeze(restored_params))
        elif train_state is None:
             # Fallback to random init if no checkpoint (debug)
             master_print("WARNING: No checkpoint loaded. Initializing random model.")
             train_state = sharded_init_fn(next_rng())

        # Compile step function
        sharded_inference_step = pjit(
            inference_step,
            in_shardings=(train_state_partition, PS(), PS()),
            out_shardings=(PS(), PS(), PS(), PS()), # rng, loss, ttt_stats, logits
        )

        # Prepare for Bucketing Analysis
        if master_process:
            tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_name)
            syntax_ids = get_syntax_ids(tokenizer)

        # Run Inference Loop
        results = []
        
        step_count = 0
        for batch in tqdm(loader, disable=not master_process, desc="Running Inference"):
            if step_count >= FLAGS.max_steps:
                break
                
            for k in batch.keys():
                batch[k] = batch[k].numpy()
            
            sharded_rng, loss, ttt_stats, logits = sharded_inference_step(train_state, sharded_rng, batch)
            
            # Gather results to host
            loss_np = jax.device_get(process_allgather(loss))
            ttt_stats_np = jax.device_get(process_allgather(ttt_stats))
            input_tokens_np = batch["input_tokens"]
            
            if master_process:
                batch_size, seq_len = input_tokens_np.shape
                
                step_metrics = {
                    "step": step_count,
                    "overall_loss": loss_np.mean(),
                    "layers": {},
                    "input_tokens": input_tokens_np,
                    "raw_grads": {}
                }
                
                # 1. Extract Raw Grads for Heatmap
                for l, layer_stats in enumerate(ttt_stats_np):
                    # layer_stats is tuple. index 4 is grad_norm
                    if len(layer_stats) > 4:
                        # Convert to standard numpy array just in case
                        step_metrics["raw_grads"][l] = np.array(layer_stats[4])

                # 2. Compute Bucketing Metrics
                for b in range(batch_size):
                    seq = input_tokens_np[b]
                    seen = set()
                    
                    # Create Masks
                    is_syntax = np.zeros(seq_len, dtype=bool)
                    is_retrieval = np.zeros(seq_len, dtype=bool)
                    is_novelty = np.zeros(seq_len, dtype=bool)
                    
                    for t, token in enumerate(seq):
                        tid = int(token)
                        if tid in syntax_ids:
                            is_syntax[t] = True
                        elif tid in seen:
                            is_retrieval[t] = True
                        else:
                            is_novelty[t] = True
                            
                        if tid not in syntax_ids:
                            seen.add(tid)

                    # Compute Metrics per Layer
                    for l, layer_stats in enumerate(ttt_stats_np):
                         if len(layer_stats) > 4:
                             # grad_norm shape is (batch_size, n_mini_batch, mini_batch_size) = (B, 128, 16)
                             # We flatten it to (B, 2048) and then pick [b]
                             
                             grad_norm_raw = step_metrics["raw_grads"][l]
                             # Flatten axis 1 and 2: (B, 128, 16) -> (B, 2048)
                             grad_norm_batch = grad_norm_raw.reshape(batch_size, -1)
                             
                             grad_norm = grad_norm_batch[b]
                             
                             # Ensure length matches
                             if len(grad_norm) != seq_len:
                                 # Should not happen given the math, but safe check
                                 if len(grad_norm) > seq_len:
                                     grad_norm = grad_norm[:seq_len]
                                 else:
                                     pad = seq_len - len(grad_norm)
                                     grad_norm = np.pad(grad_norm, (0, pad), 'edge')
                             
                             def safe_mean(arr, mask):
                                 if mask.sum() > 0:
                                     return arr[mask].mean()
                                 return 0.0
                                 
                             g_syn = safe_mean(grad_norm, is_syntax)
                             g_ret = safe_mean(grad_norm, is_retrieval)
                             g_nov = safe_mean(grad_norm, is_novelty)
                             
                             if l not in step_metrics["layers"]:
                                 step_metrics["layers"][l] = {
                                     "grad_syntax": [], "grad_retrieval": [], "grad_novelty": []
                                 }
                             step_metrics["layers"][l]["grad_syntax"].append(g_syn)
                             step_metrics["layers"][l]["grad_retrieval"].append(g_ret)
                             step_metrics["layers"][l]["grad_novelty"].append(g_nov)
                
                # Average over batch
                for l in step_metrics["layers"]:
                    for key in step_metrics["layers"][l]:
                        step_metrics["layers"][l][key] = np.mean(step_metrics["layers"][l][key])
                
                results.append(step_metrics)
            
            step_count += 1

        # Save results
        if master_process:
            output_dir = osp.join(FLAGS.exp_dir, FLAGS.exp_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = osp.join(output_dir, FLAGS.output_filename)
            master_print(f"Saving results to {output_path}...")
            with open(output_path, "wb") as f:
                pickle.dump(results, f)
            master_print("Done.")
if __name__ == "__main__":
    mlxu.run(main)
