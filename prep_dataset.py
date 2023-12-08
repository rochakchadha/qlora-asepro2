from datasets import load_dataset, Dataset
from random import randint
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

# load tokenizer from disk
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

#load original dataset from hub and save to disk
data = load_dataset("databricks/databricks-dolly-15k", split="train")
# data.save_to_disk("./datasets/")

def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Response\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

# template dataset to add prompt to each sample


def template_dataset(sample):
    sample["text"] = f"{tokenizer.bos_token}{format_dolly(sample)}{tokenizer.eos_token}"
    return sample

# apply prompt template per sample
dataset = data.map(template_dataset, remove_columns=list(data.features))
# print random sample
print(dataset[randint(0, len(dataset))]["text"])

max_length = 1024

# tokenize dataset
tok_dataset = dataset.map(lambda sample: tokenizer(sample["text"], max_length=max_length, truncation=True), batched=True, remove_columns='text')

#save tokenized dataset to disk
tok_dataset.save_to_disk("./train_dataset/")
