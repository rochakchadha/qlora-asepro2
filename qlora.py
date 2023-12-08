import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig)
from datasets import (load_dataset, Dataset)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel)


model_path = "./base_model/Mistral-7B-v0.1/"
training_path = "./train_dataset/"
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
    use_flash_attention_2=True
    quantization_config=bnb_config
)
gradient_checkpointing = True
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)  # https://github.com/huggingface/peft/blob/52ff0cde9f2cc64059e171c2cfd94512914c85df/src/peft/utils/other.py#L92

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# setup LoRA

#as per https://wandb.ai/vincenttu/finetuning_mistral7b/reports/Fine-tuning-Mistral-7B-with-W-B--Vmlldzo1NTc3MjMy
lora_modules = ['o_proj', 'up_proj', 'down_proj', 'q_proj', 'v_proj', 'gate_proj', 'k_proj']

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

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
#shuffled_dataset = dataset.shuffle(seed=42)
#eval_dataset = shuffled_dataset.select(range(300))


# define training argument

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    num_train_epochs=1,
    fp16=True,
    save_steps=75,
    logging_steps=20,
    save_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset,
    #eval_dataset=eval_dataset,
)
trainer.train()
