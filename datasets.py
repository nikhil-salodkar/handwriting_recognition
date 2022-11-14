import os
import pickle
from typing import Optional

import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader


class Kaggle_handwritten_names(Dataset):
    '''
    To solve Sequential Sentence classification problem.
    '''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_name = row['FILENAME']
        image_label = row['IDENTITY']

        return {
            'input_ids': tokenized_article['input_ids'],
            'attention_mask': tokenized_article['attention_mask'],
            'special_mask': list(special_mask),
            'targets': label_indexes
        }


class RCTSequentialSentenceDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, tokenizer, train_batch=16, val_batch=64):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_batch_size = train_batch
        self.val_batch_size = val_batch
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = RCTSequentialSentenceDataset(self.train_data, self.tokenizer)
            self.val = RCTSequentialSentenceDataset(self.val_data, self.tokenizer)

    def custom_collate(data):
        input_ids = []
        attention_mask = []
        special_mask = []
        targets = []
        for d in data:
            if d is not None:
                input_ids.append(torch.tensor(d['input_ids']))
                attention_mask.append(torch.tensor(d['attention_mask']))
                special_mask.append(torch.tensor(d['special_mask'], dtype=torch.bool))
                targets = targets + d['targets']
            else:
                continue
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        special_mask = pad_sequence(special_mask, batch_first=True, padding_value=0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'special_mask': special_mask,
            'target': torch.tensor(targets)
        }

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, shuffle=True, pin_memory=True,
                          num_workers=8, collate_fn=RCTSequentialSentenceDataModule.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, shuffle=False, pin_memory=True,
                          num_workers=8, collate_fn=RCTSequentialSentenceDataModule.custom_collate)


def test_rct():
    pl.seed_everything(567)
    with open(os.path.join('./datasets/biomedical/RCT/', 'rct_train_dict.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join('./datasets/biomedical/RCT/', 'rct_dev_dict.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join('./pretrained_models/roberta-base'))
    special_tokens_dict = {'additional_special_tokens': ['<sentence_sep>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    sample_module = RCTSequentialSentenceDataModule(train_data, val_data, tokenizer, train_batch=8)
    # the_dataset = RCTSequentialSentenceDataset(data, tokenizer)
    # return_dict = the_dataset[0]
    sample_module.setup()
    sample_batch = next(iter(sample_module.train_dataloader()))
    print(sample_batch['input_ids'].shape)
    print(sample_batch['attention_mask'].shape)
    print(sample_batch['special_mask'].shape)
    print(len(sample_batch['special_mask']))
    print(sample_batch['target'].shape)
    print(sample_batch)


if __name__ == '__main__':
    test_rct()