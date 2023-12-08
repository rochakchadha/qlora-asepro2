import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import ( PeftModel)

model_path = "./base_model/Mistral-7B-v0.1/"
checkpoint_path = "./lora_checkpoints/checkpoint-300"
merged_model_path = "./merged_model/Mistral_7b_dolly_15k"

# load the model back in 32bit as we will merge the peft model to it.

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')

# Call the tokenizer after the model, so that it's loaded on the GPU. 
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

#load the PEFT model from the last checkpoint FIX THE PATH
peft_model = PeftModel.from_pretrained(model, checkpoint_path, from_transformers=True)
model = peft_model.merge_and_unload()

#Save the merged model locally
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

def run_sample_inference():
    # Run the merged model with sample prompt
    system_instruct = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    instruction_def = " Why can camels survive for long without water?"
    prompt = "### Instruction\n" + instruction_def + "### Response\n"

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    outputs =  model.generate(
        input_ids=input_ids,
        do_sample=True,
        top_p=0.95,
        top_k=60,
        temperature=0.1,
        max_new_tokens=200,
        return_dict_in_generate = True,
        output_scores = True,)

    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    print(text[len(prompt):])

run_sample_inference() 