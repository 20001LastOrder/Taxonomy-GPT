from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification
import torch
import pytorch_lightning as pl
from embeddings import PartialOverrideEmbedding
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, precision_score, recall_score


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
        self.loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.model = get_gpt_binary_model(self.hparams)

        self.metrics = {}
        for phrase in ['train', 'val', 'test']:
            self.metrics[phrase] = {}

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self.forward(input_ids, attention_mask)
        
        # measure metrics
        loss= self.log_metric('train', logits, labels)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels=batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self.forward(input_ids, attention_mask)

        self.log_metric('val', logits, labels)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels=batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self.forward(input_ids, attention_mask)
        
        self.log_metric('test', logits, labels)

    def log_metric(self, phrase, logits, labels):
        preds = (logits > 0).int()

        # calculate the weighted loss since there are more negative pairs then positive ones
        loss = self.loss_fct(logits, labels.float())
        weights = (labels == 1).float() + (labels == 0).float() # (labels == 1).float() * self.hparams.negative_ratio + (labels == 0).float()
        loss = (loss * weights).mean()

        acc = (preds == labels).float().mean()
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)


        self.log(f'{phrase}_loss', loss)
        self.log(f'{phrase}_acc', acc)
        self.log(f'{phrase}_f1', f1)
        self.log(f'{phrase}_precision', precision)
        self.log(f'{phrase}_recall', recall)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer



def get_gpt_binary_model(hparams):
    base_model = hparams.get('base_model', 'EleutherAI/gpt-neo-1.3B')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    padding_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
    model = AutoModelForSequenceClassification.from_pretrained(base_model, pad_token_id=padding_id)
    
    for param in model.parameters():
        param.requires_grad = False
    

    # pop last layer 
    if type(model) == RobertaForSequenceClassification:
        model.classifier.out_proj = torch.nn.Linear(1024, 1)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.score = torch.nn.Linear(2048, 1)
        for param in model.score.parameters():
            param.requires_grad = True


    # modify the model based on the training configuraiton
    # update the embedding layer of the first transformer block for prompt learning
    if hparams.prompt_learning:
            new_wte = PartialOverrideEmbedding(model.transformer.wte, 25)
            model.transformer.set_input_embeddings(new_wte)

    # wrappe the model with lora using  PEFT model
    if hparams.lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none'
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # updat eth last n layers for finetuning (Prompt tuning or lora corresponds to n=0)
    print(model)

    if type(model) == RobertaForSequenceClassification:
        num_layers = len(model.roberta.encoder.layer)
        learnable_layers = hparams.learnable_layers if hparams.learnable_layers != -1 else num_layers
        for i in range(learnable_layers):
            print('Changing layer ', num_layers - 1 - i)
            layer = model.roberta.encoder.layer[num_layers - 1 - i]
            for param in layer.parameters():
                param.requires_grad = True
    else:
        num_layers = len(model.transformer.h)
        learnable_layers = hparams.learnable_layers if hparams.learnable_layers != -1 else num_layers
        for i in range(learnable_layers):
            print('Changing layer ', num_layers - 1 - i)
            for param in model.transformer.h[num_layers - 1 - i].mlp.parameters():
                param.requires_grad = True

    return model


# def load_model(model_path, is_lora=False):
#     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

#     if is_lora:
#         padding_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
#         model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id=padding_id)
    
#         PeftConfig.from_pretrained(model_path)
#     model = GPTNeoForSequenceClassificationBinary.from_pretrained(model_path)
#     return model