from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification, AutoTokenizer
import torch

def get_gpt_neo_model():
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    for param in model.parameters():
        param.requires_grad = False

    # enable the last layer
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return model


class GPTNeoForSequenceClassificationBinary(GPTNeoForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = torch.nn.Linear(2048, 1, bias=True)

def get_gpt_binary_model():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    padding_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
    model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id=padding_id)
    
    for param in model.parameters():
        param.requires_grad = False
    

    # pop last layer 
    model.score = torch.nn.Linear(2048, 1)
    for param in model.score.parameters():
        param.requires_grad = True

    return model