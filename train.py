import argparse
from transformers import TrainingArguments, Trainer

from dataset import get_taxonomy_dataset, DataCollatorWithPaddingAndMasking
from model import get_gpt_neo_model
from trainer import TaxonomyTrainer
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader


def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    datasets = get_taxonomy_dataset(args.dataset)
    model = get_gpt_neo_model()
    data_collator = DataCollatorWithPaddingAndMasking(tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        remove_unused_columns=False
    )

    trainer = TaxonomyTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=datasets['train'],         # training dataset
        eval_dataset=datasets['test'],            # evaluation dataset
        data_collator=data_collator
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/notebooks/taxonomy.csv')

    configs = parser.parse_args()
    main(configs)