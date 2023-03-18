from datasets import load_dataset
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase
import torch
from typing import List, Dict, Any, Union, Optional
from torch.nn import functional

prompt_template = "I am doing the taxonomy research. I think {child} is a subtopic of {parent}"

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
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    dataset = load_dataset('csv', data_files=filename, split='train')
    dataset = dataset.filter(lambda example: example['flag'])
    dataset = dataset.map(lambda example: format_dataset(example, tokenizer))
    dataset = dataset.remove_columns(['child', 'parent', 'Unnamed: 0', 'group', 'flag'])
    datasets = dataset.train_test_split(test_size=0.1)

    return datasets

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