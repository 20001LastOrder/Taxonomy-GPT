from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch
from typing import List, Dict, Any, Union, Optional
from torch.nn import functional
import pytorch_lightning as pl
from transformers.data.data_collator import DataCollatorWithPadding
import networkx as nx
from collections import defaultdict
import numpy as np
from datasets import Dataset, concatenate_datasets


prompt_template = "I am doing the taxonomy research. I think {child} is a subtopic of {parent}"


class PairPredictionDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.negative_ratio = hparams.negative_ratio
        self.prompt_learning = hparams.prompt_learning
        self.tokenizer = tokenizer
        self.dataset=  get_taxonomy_dataset_binary(self.hparams.dataset)
        # self.train_group = self.group_pairs(self.dataset['train'])
        # self.train_negatives = {}
        # for key, value in self.train_group.items():
        #     self.train_negatives[key] = self.create_negative_pairs(value)

    def group_pairs(self, dataset):
        result = defaultdict(list)
        for item in dataset:
            # only add positive examples
            if item['labels'] == 1:
                result[item["group"]].append(item)
        return result
    
    def create_negative_pairs(self, edges):
        res=[]
        G=nx.DiGraph()
        links = [(edge['parent'], edge['child']) for edge in edges]
        G.add_edges_from(links)
        parents = set(link[0] for link in links)
        children = set(link[1] for link in links)
        list_check = list(parents | children)
        res = []
        for i in list_check:
            desc = nx.descendants(G, i)
            all_nodes= desc
            list_check.remove(i)
            candidates = list(set(list_check)- set(all_nodes))
            res.extend([{'labels': 0, 'parent': i, 'child': candidate} for candidate in candidates])
        return res
    
    def negative_sampling(self, negatives, group, num_samples):
        idx = range(len(negatives[group]))
        # shuffle the index
        idx = np.random.choice(idx, num_samples, replace=True)
        return [negatives[group][i] for i in idx]

    def create_negative_examples(self, dataset):
        # get all groups
        group = self.group_pairs(dataset)

        negatives = {}
        for key, value in group.items():
            negatives[key] = self.create_negative_pairs(value)

        negative_examples = []
        for group, edges in group.items():
            negative_examples.extend(
                self.negative_sampling(negatives, group, len(edges) * self.negative_ratio)
            )
        
        negative_examples = Dataset.from_list(negative_examples)
        return negative_examples

    
    def format_dataset(self, example):
        prompt = prompt_template.format(child=example['child'], parent=example['parent'])

        if self.prompt_learning:
            # in prompt learning, we can give any token id as the embedding is learnt
            prefix = [-i for i in range(10)]
            middle = [-i + 10 for i in  range(5)]
            postfix = [-i + 15 for i in range(10)]
            parent_tokens = self.tokenizer.encode(example['parent'])
            child_tokens = self.tokenizer.encode(example['child'])
            example['input_ids'] = prefix + parent_tokens + middle + child_tokens + postfix
        else:
            example['input_ids'] = self.tokenizer.encode(prompt)
        return example

    def train_dataloader(self):
        dataset = self.dataset['train']
        negative_examples = self.create_negative_examples(dataset)

        dataset = concatenate_datasets([dataset, negative_examples])
        dataset = dataset.map(self.format_dataset)
        dataset = dataset.remove_columns(['child', 'parent', 'group'])

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        dataset = self.dataset['val']
        negative_examples = self.create_negative_examples(dataset)

        dataset = concatenate_datasets([dataset, negative_examples])
        dataset = dataset.map(self.format_dataset)
        dataset = dataset.remove_columns(['child', 'parent', 'group'])

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        dataset = self.dataset['test']
        negative_examples = self.create_negative_examples(dataset)

        dataset = concatenate_datasets([dataset, negative_examples])
        dataset = dataset.map(self.format_dataset)
        dataset = dataset.remove_columns(['child', 'parent', 'group'])

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            num_workers=self.hparams.num_workers,
        )
    

class DataCollatorWithPaddingAndMasking:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding: Union[bool, str] = True, max_length: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, return_tensors: str = "pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_batch = [{'input_ids': d['input_ids']} for d in features]
        mask_batch = [d['label_mask'] for d in features]
        max_length = max(len(d['input_ids']) for d in features)
        mask_batch = [functional.pad(torch.LongTensor(mask), (0, max_length - len(mask))) for mask in mask_batch]
        mask_batch = torch.stack(mask_batch)

        batch = self.tokenizer.pad(
            input_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch['label_mask'] = mask_batch
        
        return batch


def format_dataset(example, tokenizer):
    example['child'] = example['child'].replace('_', ' ')
    example['parent'] = example['parent'].replace('_', ' ')
    prompt = prompt_template.format(child=example['child'], parent=example['parent'])
    example['input_ids'] = tokenizer.encode(prompt)
    parent_tokens = tokenizer.encode(example['parent'], add_prefix_space=True)
    mask = torch.zeros(len(example['input_ids']), dtype=torch.long)
    mask[-len(parent_tokens):] = 1
    example['label_mask'] = mask
    return example

def get_taxonomy_dataset(filename):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    dataset = load_dataset('csv', data_files=filename, split='train')
    dataset = dataset.filter(lambda example: example['flag'])
    dataset = dataset.map(lambda example: format_dataset(example, tokenizer))
    dataset = dataset.remove_columns(['child', 'parent', 'Unnamed: 0', 'group', 'flag'])
    datasets = dataset.train_test_split(test_size=0.1)

    return datasets

def format_dataset_binary(example, tokenizer):
    example['child'] = example['child'].replace('_', ' ')
    example['parent'] = example['parent'].replace('_', ' ')
    prompt = prompt_template.format(child=example['child'], parent=example['parent'])
    example['input_ids'] = tokenizer.encode(prompt)

    example['labels'] = 1 # if example['flag'] else 0
    return example

def get_taxonomy_dataset_binary(filename, entire_dataset=False, remove_columns=False):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    dataset = load_dataset('csv', data_files=filename, split='train')
    dataset = dataset.map(lambda example: format_dataset_binary(example, tokenizer))

    # if "Unnamed: 0" in list(dataset.features.keys()):
    #     dataset = dataset.remove_columns(['Unnamed: 0', 'flag'])
    # else: 
    if 'type' in list(dataset.features.keys()):
        dataset = dataset.remove_columns(['type'])

    if remove_columns:
        dataset = dataset.remove_columns(['child', 'parent', 'group'])

    # shuffle the train dataset but not the test dataset
    if not entire_dataset:
        train_dataset = dataset.filter(lambda example: example['group'] < 533).shuffle()
        val_dataset = dataset.filter(lambda example: example['group'] >= 533 
                                     and example['group'] <647)
        test_dataset = dataset.filter(lambda example: example['group'] >= 647)

        # datasets = dataset.train_test_split(test_size=0.1)
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    else:
        return {
            'test': dataset
        }
