from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification, AutoTokenizer
import torch
import pytorch_lightning as pl
from embeddings import PartialOverrideEmbedding


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

        
class GPTSequenceClassifiationModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss_fct = torch.nn.BCEWithLogitsLoss()
        self.model = get_gpt_binary_model()
        if self.hparams.prompt_learning:
            new_wte = PartialOverrideEmbedding(self.model.transformer.wte, 25)
            self.model.transformer.set_input_embeddings(new_wte)

        num_layers = len(self.model.transformer.h)
        for i in range(self.hparams.learnable_layers):
            print('Changing layer ', num_layers - 1 - i)
            for param in self.model.transformer.h[num_layers - 1 - i].mlp.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fct(logits, labels.float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels=batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fct(logits, labels.float())
        preds = (logits > 0).int()
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels=batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fct(logits, labels.float())
        preds = (logits > 0).int()
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer



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
