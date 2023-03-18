from transformers import GPTNeoForCausalLM, GPT2Tokenizer


def get_gpt_neo_model():
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    for param in model.parameters():
        param.requires_grad = False

    # enable the last layer
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return model