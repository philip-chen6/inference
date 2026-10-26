import torch
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


class InferenceEngine:

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(config.model, dtype="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.padding_side = "left"
        self.config = config
        self.draft_model = None
        
        if config.spec_decode:
            self.draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto").to(self.device)
            draft_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            if self.tokenizer.get_vocab() != draft_tokenizer.get_vocab():
                raise ValueError("Target and draft token IDs must match.")
        
        print("model device: ", self.model.device)

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        model_inputs = self.tokenizer.apply_chat_template( #instruct model
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        if self.draft_model is not None:
            output = spec_dec_inference_loop(self.model, self.draft_model, self.device, self.tokenizer, model_inputs, max_new_tokens=512, num_draft_tokens=3)[0]
        else:
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
    

def cache_inference_loop(model, device, tokenizer, model_inputs, max_new_tokens=50, method="sample", temperature=0.6):
    sync()
    start = time.perf_counter()
    model.eval()

    batch_size = model_inputs["input_ids"].shape[0]
    with torch.inference_mode():
        past_key_values = None # we must manually manage key values
        #prefill step
        outputs = model(**model_inputs, past_key_values=past_key_values, use_cache=True) # use_cache=True tells pytorch to append key values to past key values
        past_key_values = outputs.past_key_values # store the past key values for the next iteration
        
        logits = outputs.logits[:, -1, :]
        if method == "sample":
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1)
            next_tokens = torch.multinomial(probs, num_samples=1) 
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True) #By default, a reduction like argmax removes the dimension it operates over. keep dim true same as unsqueeze above as model expects input of size [batch, seq_len]
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
            logits = outputs.logits[:, -1, :]
            if method == "sample":
                logits = logits / temperature
                probs = torch.softmax(logits, dim=1)
                next_tokens = torch.multinomial(probs, num_samples=1) 
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
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

def draft(draft_model, tokenizer, input_id, attention_mask, draft_cache, num_draft_tokens=5, method="sample", temperature=0.6):
    draft_model.eval()
    draft_sequence = input_id.clone() #store drafted sequence
    next_token = input_id
    draft_probs = [] #store probabilities so we can accept based off confidence
    with torch.inference_mode():
        for i in range(num_draft_tokens):
            # extend attention mask for new token
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(next_token.shape)], dim=-1)
            outputs = draft_model(next_token, attention_mask=attention_mask, past_key_values=draft_cache, use_cache=True)
            draft_cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float() # output logits has shape [batch_size, sequence_length, vocab_size]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1)
            if method == "sample":
                next_token = torch.multinomial(probs, num_samples=1) 
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)            
            
            draft_probs.append(probs[0]) # ordered by token, so you can access the probability at token_id
            draft_sequence = torch.cat([draft_sequence, next_token], dim=1)
            if next_token == tokenizer.eos_token_id:
                break
    return draft_sequence, draft_probs, draft_cache #return cache so we can keep track between drafts.
def spec_dec_inference_loop(model, draft_model, device, tokenizer, model_inputs, max_new_tokens=50, num_draft_tokens=5, method="sample", temperature=0.6):
    sync()
    start = time.perf_counter()
    model.eval()
    accepted_drafts = 0
    proposed_drafts = 0
    # batch_size = model_inputs["input_ids"].shape[0]
    with torch.inference_mode():
        past_key_values = None # we must manually manage key values
        #prefill step
        outputs = model(**model_inputs, past_key_values=past_key_values, use_cache=True) # use_cache=True tells pytorch to append key values to past key values
        
        target_cache = outputs.past_key_values # store the past key values for the next iteration
        
        logits = outputs.logits[:, -1, :].float()
        if method == "sample":
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1)
            input_tokens = torch.multinomial(probs, num_samples=1) 
        else:
            input_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        
        draft_outputs = draft_model(**model_inputs, use_cache=True)
        draft_cache = draft_outputs.past_key_values
        sync()
        first_token_time = time.perf_counter()
        
        time_drafting = 0.0
        time_verifying = 0.0
        i = 1
        while i < max_new_tokens:
            #draft 
            
            #must rebuild this every time depending on accepted token shape.
            history = torch.cat([model_inputs["input_ids"], input_tokens], dim=1)
            prompt_mask = model_inputs["attention_mask"]
            history_mask = torch.cat(
                [prompt_mask, prompt_mask.new_ones(input_tokens.shape)], dim=1
            )
            
            cached_length = draft_cache.get_seq_length()
            draft_inputs = history[:, cached_length:]
            draft_mask = history_mask[:, :cached_length]
            draft_time = time.perf_counter()
            draft_sequence, draft_probs, draft_cache = draft(draft_model, tokenizer, draft_inputs, draft_mask, draft_cache, num_draft_tokens, method, temperature)
            time_drafting += time.perf_counter() - draft_time
            # verification
            
            proposals = draft_sequence[:, draft_inputs.shape[1]:]
            k = proposals.shape[1]

            inputs = {
                "input_ids": torch.cat([history[:, -1:], proposals], dim=1),
                "attention_mask": torch.cat(
                    [history_mask, history_mask.new_ones(proposals.shape)], dim=1
                ),
            }
            verify_time = time.perf_counter()
            target_outputs = model(**inputs, past_key_values=target_cache, use_cache=True)
            
            time_verifying += time.perf_counter() - verify_time
            target_cache = target_outputs.past_key_values
            target_logits = target_outputs.logits[:, -(k+1):-1, :].float() #convert to float so that precise values aren't lost for the sampling logic
            target_probs = F.softmax(target_logits[0, :, :] / temperature, dim = -1)
            
            # acceptance/rejection check
            
            accepted_tokens = []
            proposed_drafts += k
            for j in range(k):
                draft_token = proposals[0, j].item()
                if method == "sample":
                    q = draft_probs[j]
                    p = target_probs[j]
                    
                    accept_prob = (p[draft_token] / q[draft_token]).clamp(max=1.0) #handle both p>=q and accept case of p < q
                    u = torch.rand((), device=p.device)
                    
                    if u < accept_prob: #accept with probability
                        accepted_tokens.append(draft_token)
                        accepted_drafts += 1
                        if draft_token == tokenizer.eos_token_id:
                            break
                    else:
                        new_distribution = torch.clamp(p-q, min=0.0) # as outlined in paper, this is lossless.
                        target_token = torch.multinomial(new_distribution, num_samples=1).item() #normalization auto applied in multinomial
                        accepted_tokens.append(target_token)
                        break                
                else:
                    target_token = target_logits[0, j, :].argmax().item()
                    if target_token == draft_token:
                        accepted_tokens.append(draft_token)
                        accepted_drafts += 1
                        if target_token == tokenizer.eos_token_id:
                            break
                    else:
                        accepted_tokens.append(target_token)
                        break
                
                
            # if entire draft finishes, sample extra token for free
            else:
                final_logits = target_outputs.logits[:, -1, :].float()
                if method == "sample":
                    final_logits = final_logits / temperature
                    probs = torch.softmax(final_logits, dim=1)
                    extra_token = torch.multinomial(probs, num_samples=1) 
                else:
                    extra_token = torch.argmax(final_logits, dim=-1, keepdim=True)
                accepted_tokens.append(extra_token.item())
                
            accepted_tokens = accepted_tokens[:max_new_tokens - i] #so we don't go over max
            i += len(accepted_tokens)
            
            # commit accepted tokens
            input_tokens = torch.cat([input_tokens, torch.tensor([accepted_tokens], device=device)], dim=1)
            
            if accepted_tokens[-1] == tokenizer.eos_token_id:
                break
            # get rid of rejected proposals from target cache: prompt + accepted tokens is max length
            cache_limit = model_inputs["input_ids"].shape[1] + input_tokens.shape[1] - 1 # picked token isnt in cache yet.
            # if longer, then get rid of excess.
            for cache in (draft_cache, target_cache):
                excess = cache.get_seq_length() - cache_limit
                if excess > 0:
                    cache.crop(-excess)

        sync()
        elapsed = time.perf_counter() - start
        generated = input_tokens.shape[1]

        print(f"Generation time: {elapsed:.3f}s")
        print(f"Generated tokens: {generated}")
        print(f"Throughput: {generated / elapsed:.2f} tokens/s")
        print(f"Accepted draft tokens: {accepted_drafts}")
        print(f"Acceptance rate: {accepted_drafts/proposed_drafts}")
        print(f"Drafting time: {time_drafting:.3f}s")
        print(f"Target verification time: {time_verifying:.3f}s")    
        output = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)
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
