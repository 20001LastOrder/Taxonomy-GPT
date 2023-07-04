from transformers import TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
import pandas as pd
from dataset import PairPredictionDataModule, get_taxonomy_dataset, DataCollatorWithPaddingAndMasking, get_taxonomy_dataset_binary
from model import GPTSequenceClassifiationModule, get_gpt_neo_model, get_gpt_binary_model
from trainer import TaxonomyTrainer, TaxonomyTrainerBinary
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from evaluate import load as load_metric
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from configs import get_config

training_map = {
    get_gpt_neo_model: (TaxonomyTrainer, DataCollatorWithPaddingAndMasking, get_taxonomy_dataset),
    get_gpt_binary_model: (TaxonomyTrainerBinary, DataCollatorWithPadding, get_taxonomy_dataset_binary),
}

metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = (predictions > 0).astype(int)
    return metric.compute(predictions=predictions, references=labels)

def main(args):
    model_func = get_gpt_binary_model
    Trainer = training_map[model_func][0]
    DataCollector = training_map[model_func][1]
    dataset_func = training_map[model_func][2]

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    datasets = dataset_func(args.dataset)
    model = model_func()
    data_collator = DataCollector(tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=5,              # total # of training epochs
        evaluation_strategy='steps',
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1,
        eval_steps=500,
        remove_unused_columns=False,
        save_strategy='epoch',
        report_to='wandb'
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=datasets['train'],         # training dataset
        eval_dataset=datasets['test'],            # evaluation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    predictions = trainer.predict(datasets['test'])
    df = pd.DataFrame(predictions.predictions)
    df["True_Label"] = predictions.label_ids
    df.to_csv('/notebooks/results/test.csv')

def main_lightning(args):
    if args.seed is not None:
        pl.seed_everything(args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = PairPredictionDataModule(tokenizer, args)
    model = GPTSequenceClassifiationModule(args)
    # wandb_logger = WandbLogger(project=args.project_name)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=args.run_dir, filename=args.model_name)

    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        accelerator='gpu',
        devices=-1,
        log_every_n_steps=1,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        precision=args.precision,
        reload_dataloaders_every_n_epochs=1,
        # logger=wandb_logger,
    )

    trainer.fit(model=model, datamodule=dataset)
    trainer.test(ckpt_path='last',dataloaders = dataset.test_dataloader())


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('medium')
    
    configs = get_config('config.yml')
    main_lightning(configs)