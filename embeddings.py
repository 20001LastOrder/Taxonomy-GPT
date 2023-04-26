import torch
import torch.nn as nn


class PartialOverrideEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                length_override: int = 20):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            start_override (int, optional): first token id which will be trained separately. Defaults to 110 ([unused100] for BERT).
            length_override (int, optional): how many tokens are to be trained separately after the first. Defaults to 800 ([unused900] for BERT).
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(PartialOverrideEmbedding, self).__init__()
        self.length_override = length_override
        self.wte = wte
        self.wte_override = nn.Embedding(
            length_override, wte.weight.shape[1]
        )
        # if initialize_from_vocab:
        #     self.wte_override.weight[:] = self.wte.weight[self.start_override:self.start_override+self.length_override]
        # 
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # Detect which tokens are the ones to learn
        mask_overrride = (tokens < 0) 
        mask_exist = (tokens >= 0)

        embedded_tokens = self.wte(tokens * mask_exist)

        # Zero out the ones which already have pretrained embedding and flip the sign
        modified_tokens = -tokens * mask_overrride

        # Get the
        embedded_tokens = self.wte_override(modified_tokens)

        # And finally change appropriate tokens from placeholder embedding created by
        # pretrained into trainable embeddings.
        #return embedded_tokens * torch.logical_not(mask_out) + embedded_tokens_after_override * mask_out
        embedded_tokens[mask_exist] = embedded_tokens[mask_exist]

        return embedded_tokens
