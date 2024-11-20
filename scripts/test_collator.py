from run_sft import DataCollatorForAssistantOnlyLM
from datasets import Dataset
from alignment import apply_chat_template
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch

def test_data_collator():
    
    # Initialize tokenizer (use the same one as your training)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print(tokenizer.encode("system"))
    print(tokenizer.encode("user"))
    print(tokenizer.encode("assistant"))
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    
    # Create collator instance
    collator = DataCollatorForAssistantOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        padding_free=True
    )
    
    # Create sample conversations
    sample_messages = [
        {
            "uid": "test_1",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user", 
                    "content": "Hello!"
                },
                {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                {
                    "role": "user",
                    "content": "How are you?"
                },
                {
                    "role": "assistant",
                    "content": "I'm doing great, thanks!"
                },
                {
                    "role": "user",
                    "content": "You are a helpful assistant."
                }
            ]
        },
        {
            "uid": "test_2", 
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2+2?"
                },
                {
                    "role": "assistant", 
                    "content": "2+2 equals 4."
                },
                {
                    "role": "user",
                    "content": "You are a helpful assistant."
                }
            ]
        }
    ]
    dataset = Dataset.from_list(sample_messages)
    column_names = dataset.column_names
    
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": False
        },
        num_proc=1,
        remove_columns=column_names,
        desc="Applying chat template",
    )
    
    def tokenize(element):
        outputs = tokenizer(
            element["text"], 
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=8192,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
    
    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names,
        "batch_size": 2,
        "num_proc": 1
    }
    tokenized_dataset = dataset.map(tokenize, **map_kwargs, desc="Tokenizing")
    
    # Setup DataLoader parameters
    dataloader_params = {
        "batch_size": 2,  # Small batch size for testing
        "collate_fn": collator,
        "num_workers": 0,  # No multi-processing for testing
        "pin_memory": True,
        "shuffle": False,  # Keep order deterministic for testing
    }
    
    # Create DataLoader
    dataloader = DataLoader(tokenized_dataset, **dataloader_params)
    
    # Test the first batch
    batch = next(iter(dataloader))
    
    # Add debugging prints
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape:", value.shape)
            print(f"Sample {key}:", value[0].tolist())  # Show first 50 tokens of first item
    
    # Verify the assistant responses are properly labeled
    labels = batch["labels"][0]
    mask = labels != -100
    print("\nAssistant response tokens:")
    print(tokenizer.decode(batch["input_ids"][0][mask]))
    
    # Optional: test a few more batches
    print("\nTesting additional batches...")
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Test first 3 batches
            break
        print(f"\nBatch {i+1} shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")

if __name__ == "__main__":
    test_data_collator()