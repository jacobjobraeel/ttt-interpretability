import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from ttt.dataloader.language_modeling_hf import LMDataModule
from ttt.infra.jax_utils import master_print

class QuickTestDataModule(LMDataModule):
    def __init__(self, *args, num_books=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_books = num_books

    def process_dataset(self):
        """
        Overrides the original process_dataset to load from stream
        instead of downloading the full dataset.
        """
        # 1. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        
        # 2. Stream the dataset (Validation split usually smaller/safer for testing)
        master_print(f"Streaming first {self.num_books} books from {self.dataset_name}...")
        dataset = load_dataset(
            self.dataset_name, 
            self.dataset_config_name, 
            split="validation", 
            streaming=True,
            trust_remote_code=True
        )
        
        # 3. Take subset and tokenize immediately
        subset = list(dataset.take(self.num_books))
        
        all_token_ids = []
        for i, book in enumerate(subset):
            text = book['text']
            master_print(f"Tokenizing book {i+1}/{self.num_books} (length: {len(text)} chars)...")
            
            # Tokenize
            tokens = tokenizer(text)["input_ids"]
            
            # Add EOS token if configured
            if self.add_eos:
                tokens.append(tokenizer.eos_token_id)
                
            all_token_ids.extend(tokens)

        # 4. Convert to the format LMDataset expects (numpy array)
        master_print(f"Tokenized {len(all_token_ids)} tokens.")
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
        full_ids = np.array(all_token_ids, dtype=dtype)
        
        # 5. Return in the dictionary format expected by setup()
        # Since we are only doing inference (eval_mode=True), we only populate 'validation'
        # The trainer will use 'val_dataloader()' which reads from 'validation' key
        concat_ids = {
            "train": np.array([], dtype=dtype),      # Empty, not needed for inference
            "validation": full_ids,                  # The actual data
            "test": np.array([], dtype=dtype),       # Empty
        }
        
        return concat_ids, tokenizer
