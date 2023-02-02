import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from datasets import KaggleHandwritingDataModule
from training_modules import HandwritingRecogTrainModule


def get_data(path):
    """function to return train and validation for training by reading already processed and clean dataset"""
    train_df = pd.read_csv(os.path.join(path, 'train_new.csv'))
    val_df = pd.read_csv(os.path.join(path, 'val_new.csv'))
    train_df = train_df[train_df.IDENTITY != 'UNREADABLE']
    val_df = val_df[val_df.IDENTITY != 'UNREADABLE']
    train_df = train_df[train_df.IDENTITY != 'EMPTY']
    val_df = val_df[val_df.IDENTITY != 'EMPTY']
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df, val_df


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-loss:.3f}-{val-exact-match}-{val-char-error-rate}',
                                          save_top_k=1, monitor='val-char-error-rate', mode='min', save_last=True)
    wandb_logger = WandbLogger(project="handwriting_recognition_kaggle", save_dir='./lightning_logs',
                               name='CNNR_run_new_version')
    early_stopping = EarlyStopping(monitor="val-char-error-rate", patience=10, verbose=False, mode="min")
    model_summary = ModelSummary(max_depth=-1)

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=100,
                         callbacks=[checkpoint_callback, early_stopping, model_summary], logger=wandb_logger,
                         precision=16)
    trainer.fit(train_module, data_module)


def test_handwriting_recognition():
    pl.seed_everything(15798)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-4, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition', 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 1024, 'input_height': 36, 'input_width': 324, 'gru_hidden_size': 128,
        'gru_num_layers': 2, 'num_classes': 28
    }
    label_to_index = {' ': 0, '-': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11,
                      'K': 12, 'L': 13, 'M': 14, 'N': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22,
                      'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27}
    index_to_labels = {0: ' ', 1: '-', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E', 7: 'F', 8: 'G', 9: 'H', 10: 'I',
                       11: 'J', 12: 'K', 13: 'L', 14: 'M', 15: 'N', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S',
                       21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}
    train_df, val_df = get_data(hparams['data_path'])
    data_module = KaggleHandwritingDataModule(train_df, val_df, hparams, label_to_index)
    train_module = HandwritingRecogTrainModule(hparams, index_to_labels=index_to_labels, label_to_index=label_to_index)
    train_model(train_module, data_module)


if __name__ == '__main__':
    test_handwriting_recognition()
