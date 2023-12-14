from datasets import load_dataset, Dataset
from random import randint
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

# load tokenizer from disk
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    )
tokenizer.pad_token = tokenizer.eos_token

#load original dataset from hub and save to disk
data = load_dataset("./datasets/databricks-dolly-15k", split="train")
data.save_to_disk("./datasets/")

def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Response\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

# check the distribution of the dataset. First we tokenize and then we check the length of the tokenized prompt

def token_length(example):
    result = tokenizer(format_dolly(example),return_tensors="pt", return_attention_mask=False)
    length_of_sequence = result.input_ids.shape[1]
    return length_of_sequence

# Define the buckets
buckets = {
    "<100": 0,
    "100-256": 0,
    "256-512": 0,
    "512-1024": 0,
    ">1024": 0
}

# Function to determine the bucket for a given length
def determine_bucket(length):
    if length < 100:
        return "<100"
    elif length <= 256:
        return "100-256"
    elif length <= 512:
        return "256-512"
    elif length <= 1024:
        return "512-1024"
    else:
        return ">1024"

    # Iterate over the dataset and count
    counter = 0
    for example in data:
            length = token_length(example)
            bucket = determine_bucket(length)
            buckets[bucket] += 1
            counter += 1

    # Display the distribution
    print("Distribution of sequence lengths:")
    for bucket, count in buckets.items():
        print(f"{bucket}: {count}")



