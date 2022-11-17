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
            nn.Conv2d(128, 256, stride=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.image_feature_extractor(x)


class HandwritingRecognitionGRU(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.gru_layer = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        recurrent_output, _ = self.gru_layer(X)
        out = self.output(recurrent_output)
        return out


class HandwritingRecognition(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn_feature_extractor = HandwritingRecognitionCNN()
        self.gru = HandwritingRecognitionGRU(256, 1024, 2, num_classes+1)
        self.linear = nn.Linear(1024, 256)
        self.criterion = nn.CTCLoss()

    def forward(self, X, y):
        out = self.cnn_feature_extractor(X)
        print("The output after feature extraction:", out.shape)
        batch_size, channels, width, height = out.size()
        print(batch_size, channels, width, height)
        out = out.view(batch_size, -1, height)
        print("after reshape", out.shape)
        out = out.permute(0, 2, 1)
        print("after permute", out.shape)
        out = self.linear(out)
        print("after linear", out.shape)
        out = self.gru(out)
        print("after gru")
        N = out.size(0)
        T = out.size(1)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
        target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
        print("input lengths are:", input_lengths)
        print("target lengths are:", target_lengths)
        loss = self.criterion(out, y, input_lengths, target_lengths)



def test_modelling():
    pl.seed_everything(2488)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition',
        'train_batch_size': 64, 'val_batch_size': 1024, 'input_height': 128, 'bottleneck_dim': 64, 'gru_hidden_size': 128,
        'gru_num_layers': 2, 'num_classes': 29
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
    model = HandwritingRecognition(hparams['num_classes'])
    output = model(sample_train_batch['transformed_image'])
    print(output.shape)
    # print("starting gru..")
    # model_gru = HandwritingRecognitionGRU(4, hparams['bottleneck_dim'], hparams['gru_hidden_size'],
    #                                       hparams['gru_num_layers'], hparams['num_classes'])
    # final_output = model_gru(output)
    # print("The final_output shape:", final_output.shape)
    # print(final_output)

if __name__ == '__main__':
    test_modelling()
