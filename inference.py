import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import ( PeftModel)

merged_model_path = "./merged_model/Mistral_7b_dolly_15k"

# load the model back in 32bit as we will merge the peft model to it.

model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16, device_map='auto')

# Call the tokenizer after the model, so that it's loaded on the GPU. 
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
tokenizer.pad_token = tokenizer.eos_token

def run_sample_inference():
    system_instruct = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    instruction_def = " Write ten tricks to make perfect omlette"
    prompt = "### Instruction\n" + instruction_def + "### Response\n"

    # Generate inputs and attention mask
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Generate output
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        top_p=0.35,
        top_k=50,
        temperature=0.1,
        max_length=input_ids.shape[1] + 500,  # Adjust max_length as needed
        eos_token_id=tokenizer.eos_token_id,  # End of sequence token
        pad_token_id=tokenizer.eos_token_id,  # Pad token
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        output_scores=True,
    )

    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    def post_process_response(text):
        # Split the text into sections
        sections = text.split("### Instructions")   
        # Return the first section, assuming it contains the main response
        return sections[0].strip() if sections else text
    
        # After generating the text in your function
    processed_text = post_process_response(text)
    print(processed_text)
    
run_sample_inference() 