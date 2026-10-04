import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

print("model device: ", model.device)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]

model_inputs = tokenizer.apply_chat_template( #instruct model
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
).to(device)

print("model inputs: ", model_inputs)
prompt_length = model_inputs["input_ids"].shape[1]
model.eval()
max_new_tokens = 50
with torch.inference_mode(): # more optimal than torch.no_grad() for inference
    for i in range(max_new_tokens):
        outputs = model(**model_inputs) # preferred to pass in the model_inputs dict directly instead of letting the model generate the attention_mask separately
        next_token_logits = outputs.logits[:, -1, :] # get the logits for the last token
        next_token = torch.argmax(next_token_logits, dim=-1) # dim = -1 for last dimension, corresponding to vocab
        print("next token: ", next_token)
        model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], next_token.unsqueeze(-1)], dim=-1) # append the next token to the input_ids sequence dimension. must unsqueeze since next_token is 1D and input_ids is 2D
        model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], torch.ones((1, 1), device=device)], dim=-1) # append a 1 to the attention_mask sequence dimension to have it grow.
        if next_token == tokenizer.eos_token_id:
            break



generated_ids = model_inputs["input_ids"][:, prompt_length:]
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("output: ", output)

