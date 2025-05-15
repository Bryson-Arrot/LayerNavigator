import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, tokenizer):
        super().__init__()
        self.block = block
        self.tokenizer = tokenizer

        self.activations = None
        self.add_activations = None

        self.save_new_tok_proj = False
        self.new_tok_proj = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0].detach().clone()

        if self.add_activations is not None:
            # Save New Token Projection
            if self.save_new_tok_proj:
                last_acts = self.activations[:,-1,:].detach().cpu()
                norm_vec = self.add_activations/self.add_activations.norm().item()
                proj = last_acts.matmul(norm_vec.cpu()).squeeze()
                self.new_tok_proj.append(proj)

            augmented_output = add_vector_at_last_token(
                matrix = output[0],
                vector = self.add_activations,
            )  
            output = (augmented_output,) + output[1:]

        return output
    
    def reset(self):
        self.activations = None
        self.add_activations = None

        self.save_new_tok_proj = False
        self.new_tok_proj = []
    
    
    
def add_vector_at_last_token(matrix,vector):

    batch_size, seq_len, hidden_size = matrix.shape
    mask = t.zeros(batch_size, seq_len, dtype=t.bool).to(matrix.device)
    mask[:,-1] = True
    mask =  mask.unsqueeze(-1)

    matrix += mask.float()*vector

    return matrix




class LlamaWrapper:
    def __init__(
        self,
        model_path: str,
    ):
        if t.cuda.is_available():
            self.device = "cuda"
        elif t.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_path,
                                                          pad_token_id = self.tokenizer.eos_token_id, 
                                                          eos_token_id = self.terminators
                                                          ).to(self.device)
        self.model.eval()
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.tokenizer)
        

    def generate(self, input, max_new_tokens=5):
        with t.no_grad():
            generated = self.model.generate(
                **input,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.terminators,
            )

            return self.__remove_pad_decode(generated)

    def get_logits(self,input):
        with t.no_grad():
            logits = self.model(**input).logits
            return logits
    
    def get_last_logits(self,input):
        with t.no_grad():
            logits = self.model(**input).logits
            logits = logits[:,-1,:].squeeze()
            return logits
        
    def get_activations(self, layer):
        return self.model.model.layers[layer].activations
    
    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
    
    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add_activations = activations
    
    def __remove_pad_decode(self,generated):
        decoded_texts = []
        for tokens in generated:
            filtered_tokens = [token for token in tokens if token !=self.tokenizer.pad_token_id]
            decoded_texts.append(self.tokenizer.decode(filtered_tokens, skip_special_tokens=False,clean_up_tokenization_space=True))
        return decoded_texts
    
class QwenWrapper:
    def __init__(
        self,
        model_path: str,
    ):
        if t.cuda.is_available():
            self.device = "cuda"
        elif t.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_path,
                                                          pad_token_id = self.tokenizer.eos_token_id, 
                                                          eos_token_id = self.terminators,
                                                          torch_dtype="auto",
                                                          trust_remote_code=True,
                                                          use_safetensors=True,
                                                          local_files_only=True,
                                                          low_cpu_mem_usage=True,
                                                          ).to(self.device)
        print("Model loaded")
        # self.model = self.model.to(t.float16)
        # self.model.config.sliding_window = None
        self.model.eval()
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.tokenizer)
        

    def generate(self, input, max_new_tokens=5):
        with t.no_grad():
            generated = self.model.generate(
                **input,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.terminators,
            )

            return self.__remove_pad_decode(generated)


    def get_logits(self,input):
        with t.no_grad():
            logits = self.model(**input).logits
            return logits
    
    def get_last_logits(self,input):
        with t.no_grad():
            logits = self.model(**input).logits
            logits = logits[:,-1,:].squeeze()
            return logits
        
    def get_activations(self, layer):
        return self.model.model.layers[layer].activations
    
    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
    
    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add_activations = activations
    
    def __remove_pad_decode(self,generated):
        decoded_texts = []
        for tokens in generated:
            filtered_tokens = [token for token in tokens if token !=self.tokenizer.pad_token_id]
            decoded_texts.append(self.tokenizer.decode(filtered_tokens, skip_special_tokens=False,clean_up_tokenization_space=True))
        return decoded_texts
    