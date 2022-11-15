import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale


class Kaggle_handwritten_names(Dataset):
    def __init__(self, data, transforms, img_path):
        self.data = data
        self.transforms = transforms
        self.img_path = img_path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_name = row['FILENAME']
        image_label = row['IDENTITY']
        the_image = Image.open(os.path.join(self.img_path, file_name))
        transformed_image = self.transforms(the_image)
        label_seq = [image_label]
        return {
            'transformed_image': transformed_image,
            'label': label_seq
        }


class KaggleHandwritingDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, hparams):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_batch_size = hparams['train_batch_size']
        self.val_batch_size = hparams['val_batch_size']
        self.transforms = Compose([Resize((hparams['input_height'], hparams['input_height'])), Grayscale(),
                                   ToTensor()])
        self.train_img_path = hparams['train_img_path']
        self.val_img_path = hparams['val_img_path']

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = Kaggle_handwritten_names(self.train_data, self.transforms, self.train_img_path)
            self.val = Kaggle_handwritten_names(self.val_data, self.transforms, self.val_img_path)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, shuffle=True, pin_memory=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, shuffle=False, pin_memory=True,
                          num_workers=8)


def test_kaggle_handwritting():
    pl.seed_everything(567)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition',
        'train_batch_size': 64, 'val_batch_size': 1024, 'input_height': 128
    }
    train_df = pd.read_csv(os.path.join(hparams['data_path'], 'train_new.csv'))
    train_df = train_df[train_df.word_type == 'normal_word']
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(os.path.join(hparams['data_path'], 'val_new.csv'))
    val_df = val_df[val_df.word_type == 'normal_word']
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    sample_module = KaggleHandwritingDataModule(train_df, val_df, hparams)
    sample_module.setup()
    sample_train_module = sample_module.train_dataloader()
    sample_val_module = sample_module.val_dataloader()
    sample_train_batch = next(iter(sample_train_module))
    sample_val_batch = next(iter(sample_val_module))
    print(sample_train_batch['transformed_image'].shape)
    print(sample_val_batch['transformed_image'].shape)
    print(sample_train_batch)
    print("sample_val_batch is:", sample_val_batch)



if __name__ == '__main__':
    test_kaggle_handwritting()