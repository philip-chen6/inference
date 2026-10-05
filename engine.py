import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def standard_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=50):

    model.eval() # ensures mathematical operations are correct, e.g. dropout is disabled
    prompt_length = model_inputs["input_ids"].shape[1]

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
    return output
    

def cache_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=50):

    model.eval()
    with torch.inference_mode():
        past_key_values = None # we must manually manage key values
        #prefill step
        outputs = model(**model_inputs, past_key_values=past_key_values, use_cache=True) # use_cache=True tells pytorch to append key values to past key values

        past_key_values = outputs.past_key_values # store the past key values for the next iteration
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True) #By default, a reduction like argmax removes the dimension it operates over. keep dim true same as unsqueeze above as model expects input of size [batch, seq_len]
        print("next token: ", next_token)
        generated_tokens = [next_token.item()] # store the generated token for the first iteration

        for i in range(max_new_tokens - 1): # we already generated one token, so we only need to generate max_new_tokens - 1 more
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True) # pass in the next token and the past key values
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True) # keepdim=True to maintain the shape of the tensor for concatenation
            print("next token: ", next_token)
            generated_tokens.append(next_token.item())

            if next_token == tokenizer.eos_token_id:
                break

    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return output


def main():
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

    output = cache_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=50)
    print("output: ", output)


if __name__ == "__main__":
    main()
