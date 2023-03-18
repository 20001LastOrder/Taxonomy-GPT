from torch import nn
from transformers import Trainer

class TaxonomyTrainer(Trainer):
    loss_fct = nn.CrossEntropyLoss(reduce=None)

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        masks = inputs.pop("label_mask")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        labels = inputs['input_ids'][:, 1:]
        labels = labels.reshape(-1)
        masks = masks[:, 1:].float()
        masks = masks.reshape(-1)
        
        loss = self.loss_fct(logits, labels)

        loss = (loss * masks).mean()
        return (loss, outputs) if return_outputs else loss