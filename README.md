This is a small demo for q-LoRa using https://huggingface.co/mistralai/Mistral-7B-v0.1 and converting the https://huggingface.co/datasets/databricks/databricks-dolly-15k to an instruct-response dataset. 
The intent is to convert the base Mistral model to a instruction following model. 
Prompt style: 
   <t>### Instruction\n{sample['instruction']} ### Context\n{sample['context']} ### Response\n{sample['response']}"</t>
   
