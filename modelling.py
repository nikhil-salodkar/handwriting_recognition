import os
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Func
import pytorch_lightning as pl
import torch.distributions as td
from datasets import KaggleHandwritingDataModule


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class HandwritingRecognitionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, stride=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            PrintLayer(),
            nn.Conv2d(16, 32, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            PrintLayer(),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PrintLayer(),
            nn.Conv2d(64, 128, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, stride=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.image_feature_extractor(x)


class HandwritingRecognitionGRU(nn.Module):
    def __init__(self, cnn_output_height, bottleneck_dim, gru_hidden_size, gru_num_layers, num_classes):
        super().__init__()
        self.gru_input_size = cnn_output_height * bottleneck_dim
        self.gru_layer = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers,
                          batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, image_features):
        batch_size = image_features.shape[0]
        reshaped_features = image_features.view(batch_size, -1, self.gru_input_size)
        out, _ = self.gru_layer(reshaped_features)
        return out

def test_modelling():
    pl.seed_everything(657)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition',
        'train_batch_size': 64, 'val_batch_size': 1024, 'input_height': 128, 'bottleneck_dim': 64, 'gru_hidden_size': 128,
        'gru_num_layers': 2, 'num_classes': 30
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
    model = HandwritingRecognitionCNN()
    output = model(sample_train_batch['transformed_image'])
    print(output.shape)
    print("starting gru..")
    model_gru = HandwritingRecognitionGRU(4, hparams['bottleneck_dim'], hparams['gru_hidden_size'],
                                          hparams['gru_num_layers'], hparams['num_classes'])
    final_output = model_gru(output)
    print("The final_output shape:", final_output.shape)
    print(final_output)

if __name__ == '__main__':
    test_modelling()
