import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig)
from datasets import (load_dataset, Dataset)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel)
import wandb 
import os
from datetime import datetime

model_path = "./base_model/Mistral-7B-v0.1/"
training_path = "./train_dataset/"
eval_path = "./eval_dataset/"
output_dir = "./lora_checkpoints"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config
)
gradient_checkpointing = True
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)  # https://github.com/huggingface/peft/blob/52ff0cde9f2cc64059e171c2cfd94512914c85df/src/peft/utils/other.py#L92

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="Mistral Qlora"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
# setup LoRA

#as per https://wandb.ai/vincenttu/finetuning_mistral7b/reports/Fine-tuning-Mistral-7B-with-W-B--Vmlldzo1NTc3MjMy
lora_modules = ['o_proj', 'up_proj', 'down_proj', 'q_proj', 'v_proj', 'gate_proj', 'k_proj', 'lm_head']

lora_alpha = 64
lora_dropout = 0.05
lora_r = 16

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_modules,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#load dataset from disk and define eval dataset
dataset = Dataset.load_from_disk(training_path)
eval_dataset = Dataset.load_from_disk(eval_path)

#shuffled_dataset = dataset.shuffle(seed=42)
#eval_dataset = shuffled_dataset.select(range(300))

project = "Dolly-Qlora"
base_model_name = "mistral"
run_name = base_model_name + "-" + project

# define training argument

training_arguments = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=1,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    num_train_epochs=1,
    fp16=True,
    save_steps=75,
    logging_steps=20,
    report_to="wandb",
    evaluation_strategy="steps",
    do_eval=True,
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    save_strategy="steps"
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
