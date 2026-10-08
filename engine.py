import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


class InferenceEngine:

    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        
        print("model device: ", self.model.device)

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        model_inputs = self.tokenizer.apply_chat_template( #instruct model
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
                
        output = cache_inference_loop(self.model, self.device, self.tokenizer, model_inputs, max_new_tokens=512)[0]
        print("output: ", output)
        return output

    def batch_generate(self, prompts):
        #create list of chat messages for each prompt
        messages = [
                    [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}] for prompt in prompts
        ]
        #apply chat template to each without tokenizing
        message_templates = [self.tokenizer.apply_chat_template(message, 
            tokenize=False, add_generation_prompt=True) for message in messages]

        #batch tokenize together
        model_inputs = self.tokenizer(message_templates, padding=True, return_tensors="pt").to(self.device)
        outputs = cache_inference_loop(self.model, self.device, self.tokenizer, model_inputs, max_new_tokens = 512)
        for i, output in enumerate(outputs, start=1):
            print(f"output {i}: {output}")
        return outputs
    
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def standard_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=50):
    model_inputs = {
        key: value.clone()
        for key, value in model_inputs.items()
    }    
    model.eval() # ensures mathematical operations are correct, e.g. dropout is disabled
    prompt_length = model_inputs["input_ids"].shape[1]
    sync()
    start = time.perf_counter()
    with torch.inference_mode(): # more optimal than torch.no_grad() for inference

        outputs = model(**model_inputs, use_cache=False)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True) 
        model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], next_token], dim=-1) # append the next token to the input_ids sequence dimension. must unsqueeze since next_token is 1D and input_ids is 2D
        model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], torch.ones((1, 1), device=device)], dim=-1) # append a 1 to the attention_mask sequence dimension to have it grow.
        generated_tokens = [next_token.item()]
        sync()
        first_token_time = time.perf_counter()
        for i in range(max_new_tokens - 1):
            outputs = model(**model_inputs, use_cache=False) # preferred to pass in the model_inputs dict directly instead of letting the model generate the attention_mask separately
            next_token_logits = outputs.logits[:, -1, :] # get the logits for the last token
            next_token = torch.argmax(next_token_logits, dim=-1) # dim = -1 for last dimension, corresponding to vocab
            model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], next_token.unsqueeze(-1)], dim=-1) # append the next token to the input_ids sequence dimension. must unsqueeze since next_token is 1D and input_ids is 2D
            model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], torch.ones((1, 1), device=device)], dim=-1) # append a 1 to the attention_mask sequence dimension to have it grow.
            generated_tokens.append(next_token.item())
            if next_token == tokenizer.eos_token_id:
                break
    sync()
    end = time.perf_counter()

    ttft = first_token_time - start
    tps = (len(generated_tokens) - 1) / (end - first_token_time) # typically tps ignores first token time as that is part of prefill
    total_time = end - start
    throughput = len(generated_tokens) / total_time
    print(f"Start: {start:.4f}")
    print(f"First token: {first_token_time:.4f}")
    print(f"End: {end:.4f}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Tokens generated: {len(generated_tokens)}")
    print(f"Time to first token: {ttft:.4f} seconds")
    print(f"Tokens per second: {tps:.4f}")
    print(f"Throughput: {throughput:.4f} tokens/second")
    # generated_ids = model_inputs["input_ids"][:, prompt_length:]
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return output
    

def cache_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=50):
    sync()
    start = time.perf_counter()
    model.eval()

    batch_size = model_inputs["input_ids"].shape[0]
    with torch.inference_mode():
        past_key_values = None # we must manually manage key values
        #prefill step
        outputs = model(**model_inputs, past_key_values=past_key_values, use_cache=True) # use_cache=True tells pytorch to append key values to past key values

        past_key_values = outputs.past_key_values # store the past key values for the next iteration
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True) #By default, a reduction like argmax removes the dimension it operates over. keep dim true same as unsqueeze above as model expects input of size [batch, seq_len]
        generated_tokens = next_tokens.tolist() # store generated tokens as list of lists to account for batch size
        # Track which sequences in the batch have hit EOS
        finished = next_tokens.squeeze(-1).eq(tokenizer.eos_token_id)
        attention_mask = model_inputs["attention_mask"]

        next_tokens = next_tokens.masked_fill(finished.unsqueeze(-1), tokenizer.pad_token_id) # replace "finished" sequences with pad token to avoid recording further tokens for them
        sync()
        first_token_time = time.perf_counter()
        for i in range(max_new_tokens - 1): # we already generated one token, so we only need to generate max_new_tokens - 1 more
            if finished.all().item(): # if all sequences in the batch have hit EOS, we can stop generating
                break

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1) # append a 1 to the attention_mask sequence dimension to have it grow with new tokens.
            
            outputs = model(next_tokens, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True) # pass in the next token and the past key values
            past_key_values = outputs.past_key_values
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True) # keepdim=True to maintain the shape of the tensor for concatenation
            token_ids = next_tokens.squeeze(-1)
            token_values = token_ids.tolist()
            already_finished = finished.tolist()
            for b in range(batch_size):
                if not already_finished[b]: # if the sequence has not finished, append the next token to the generated tokens
                    generated_tokens[b].append(token_values[b])
            
            finished = finished | token_ids.eq(tokenizer.eos_token_id) # update finished sequences AFTER appending the token.

            next_tokens = next_tokens.masked_fill(finished.unsqueeze(-1), tokenizer.pad_token_id) 

    sync()
    end = time.perf_counter()

    tokens_per_request = [len(ids) for ids in generated_tokens]
    total_tokens = sum(tokens_per_request)
    ttft = first_token_time - start
    tps = (total_tokens - batch_size) / (end - first_token_time)
    total_time = end - start
    throughput = total_tokens / total_time
    print(f"Start: {start:.4f}")
    print(f"First token: {first_token_time:.4f}")
    print(f"End: {end:.4f}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Tokens generated: {total_tokens}")
    print(f"Time to first token: {ttft:.4f} seconds")
    print(f"Tokens per second: {tps:.4f}")
    print(f"Throughput: {throughput:.4f} tokens/second")
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    print("model device: ", model.device)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how LLM inference works in detail"},
    ]

    model_inputs = tokenizer.apply_chat_template( #instruct model
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)

    print("model inputs: ", model_inputs)

    output = standard_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=512)
    print("output: ", output)


if __name__ == "__main__":
    main()
