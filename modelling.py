import os
from torchvision.transforms import Compose, Resize, ToTensor
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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
            nn.Conv2d(1, 32, stride=(1, 2), kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # PrintLayer(),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # PrintLayer(),
            nn.Conv2d(64, 128, stride=2, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # PrintLayer(),
            nn.Conv2d(128, 256, stride=(1, 2), kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # PrintLayer()
        )

    def forward(self, x):
        return self.image_feature_extractor(x)


class HandwritingRecognitionGRU(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.gru_layer = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        recurrent_output, _ = self.gru_layer(x)
        out = self.output(recurrent_output)
        out = F.log_softmax(out, dim=2)
        return out


class HandwritingRecognition(nn.Module):
    def __init__(self, gru_input_size, gru_hidden, gru_layers, num_classes):
        super().__init__()
        self.cnn_feature_extractor = HandwritingRecognitionCNN()
        self.gru = HandwritingRecognitionGRU(gru_input_size, gru_hidden, gru_layers, num_classes+1)
        self.linear1 = nn.Linear(1280, 512)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256)

    def forward(self, x):
        out = self.cnn_feature_extractor(x)
        batch_size, channels, width, height = out.size()
        # print(batch_size, channels, height, width)
        out = out.view(batch_size, -1, height)
        out = out.permute(0, 2, 1)
        # print(out.shape)
        out = self.linear1(out)
        # print(out.shape)
        out = self.activation(self.linear2(out))
        # print(out.shape)
        out = self.gru(out)
        # print(out.shape)
        out = out.permute(1, 0, 2)
        return out


def test_modelling():
    pl.seed_everything(6579)
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
    sample_module = KaggleHandwritingDataModule(train_df, val_df, hparams, label_to_index)
    sample_module.setup()
    sample_train_module = sample_module.train_dataloader()
    sample_val_module = sample_module.val_dataloader()
    sample_train_batch = next(iter(sample_train_module))
    sample_val_batch = next(iter(sample_val_module))
    # cnn_model = HandwritingRecognitionCNN()
    # cnn_out = cnn_model(sample_train_batch['transformed_images'])
    # print('cnn_output:', cnn_out.shape)
    model = HandwritingRecognition(hparams['gru_input_size'], hparams['gru_hidden_size'],
                           hparams['gru_num_layers'], hparams['num_classes'])
    output = model(sample_train_batch['transformed_images'])
    print("the output shape:", output.shape)
    # output = model(sample_train_batch['transformed_images'])
    # print(output.shape)
    # print("starting gru..")
    # model_gru = HandwritingRecognitionGRU(4, hparams['bottleneck_dim'], hparams['gru_hidden_size'],
    #                                       hparams['gru_num_layers'], hparams['num_classes'])
    # final_output = model_gru(output)
    # print("The final_output shape:", final_output.shape)
    # print(final_output)

if __name__ == '__main__':
    test_modelling()
