import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomRotation, RandomApply, \
    GaussianBlur, CenterCrop


class KaggleHandwrittenNames(Dataset):
    def __init__(self, data, transforms, label_to_index, img_path):
        self.data = data
        self.transforms = transforms
        self.img_path = img_path
        self.label_to_index = label_to_index

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_name = row['FILENAME']
        image_label = row['IDENTITY']
        the_image = Image.open(os.path.join(self.img_path, file_name))
        transformed_image = self.transforms(the_image)
        target_len = len(image_label)
        label_chars = list(image_label)
        image_label = torch.tensor([self.label_to_index[char] for char in label_chars])
        return {
            'transformed_image': transformed_image,
            'label': image_label,
            'target_len': target_len
        }


class KaggleHandwritingDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, hparams, label_to_index):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_batch_size = hparams['train_batch_size']
        self.val_batch_size = hparams['val_batch_size']
        self.transforms = Compose([Resize((hparams['input_height'], hparams['input_width'])), Grayscale(),
                                   ToTensor()])
        applier1 = RandomApply(transforms=[RandomRotation(10), GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5)
        applier2 = RandomApply(transforms=[CenterCrop((hparams['input_height'] - 1, hparams['input_width'] - 2))], p=0.5)
        self.train_transforms = Compose([applier2, Resize((hparams['input_height'], hparams['input_width'])), Grayscale(),
                                   applier1, ToTensor()])
        self.train_img_path = hparams['train_img_path']
        self.val_img_path = hparams['val_img_path']
        self.label_to_index = label_to_index

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = KaggleHandwrittenNames(self.train_data, self.train_transforms, self.label_to_index, self.train_img_path)
            self.val = KaggleHandwrittenNames(self.val_data, self.transforms, self.label_to_index, self.val_img_path)

    def custom_collate(data):
        '''
        To handle variable max seq length batch size
        '''
        transformed_images = []
        labels = []
        target_lens = []
        for d in data:
            transformed_images.append(d['transformed_image'])
            labels.append(d['label'])
            target_lens.append(d['target_len'])
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        transformed_images = torch.stack(transformed_images)
        target_lens = torch.tensor(target_lens)
        return {
            'transformed_images': transformed_images,
            'labels': batch_labels,
            'target_lens': target_lens
        }

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, shuffle=True, pin_memory=True,
                          num_workers=8, collate_fn=KaggleHandwritingDataModule.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, shuffle=False, pin_memory=True,
                          num_workers=8, collate_fn=KaggleHandwritingDataModule.custom_collate)


def test_kaggle_handwritting():
    pl.seed_everything(267)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition', 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256, 'input_height': 36, 'input_width': 324, 'gru_hidden_size': 128,
        'gru_num_layers': 1, 'num_classes': 28
    }
    label_to_index = {' ': 0, '-': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11,
                      'K': 12, 'L': 13, 'M': 14, 'N': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22,
                      'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27}

    train_df = pd.read_csv(os.path.join(hparams['data_path'], 'train_new.csv'))
    train_df = train_df[train_df.word_type == 'normal_word']
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(os.path.join(hparams['data_path'], 'val_new.csv'))
    val_df = val_df[val_df.word_type == 'normal_word']
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    # sample_transforms = Compose([Resize((hparams['input_height'], hparams['input_height'])), Grayscale(),
    #                                ToTensor()])
    # sample_data = KaggleHandwrittenNames(train_df, sample_transforms, label_to_index, hparams['train_img_path'])
    # print(sample_data[0], sample_data[456])
    sample_module = KaggleHandwritingDataModule(train_df, val_df, hparams, label_to_index)
    sample_module.setup()
    sample_train_module = sample_module.train_dataloader()
    sample_val_module = sample_module.val_dataloader()
    sample_train_batch = next(iter(sample_train_module))
    sample_val_batch = next(iter(sample_val_module))
    print(sample_train_batch['transformed_images'].shape)
    print(sample_val_batch['transformed_images'].shape)
    print(sample_train_batch['labels'].shape)
    print(sample_val_batch['labels'].shape)
    print(sample_train_batch['target_lens'].shape)
    print(sample_val_batch['target_lens'].shape)


if __name__ == '__main__':
    test_kaggle_handwritting()