import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from datasets import KaggleHandwritingDataModule
from training_modules import HandwritingRecogTrainModule


def get_data(path):
    train_df = pd.read_csv(os.path.join(path, 'train_new.csv'))
    val_df = pd.read_csv(os.path.join(path, 'val_new.csv'))
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df, val_df


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-loss:.3f}', save_top_k=2,
                                          monitor='val-loss', mode='min', save_last=True)
    wandb_logger = WandbLogger(project="handwriting_recognition_kaggle", save_dir='./lightning_logs', name='CNNR_arch1')
    early_stopping = EarlyStopping(monitor="val-loss", patience=10, verbose=False, mode="min")
    model_summary = ModelSummary(max_depth=-1)
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    # trainer = pl.Trainer(accelerator='gpu', fast_dev_run=True, max_epochs=200,
    #                      callbacks=[checkpoint_callback, early_stopping], precision=16)

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=50,
                         callbacks=[checkpoint_callback, early_stopping, model_summary], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


def test_handwriting_recognition():
    pl.seed_everything(15798)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition', 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256, 'input_height': 128, 'gru_hidden_size': 128,
        'gru_num_layers': 1, 'num_classes': 28
    }
    label_to_index = {' ': 0, '-': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11,
                      'K': 12, 'L': 13, 'M': 14, 'N': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22,
                      'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27}
    train_df, val_df = get_data(hparams['data_path'])
    data_module = KaggleHandwritingDataModule(train_df, val_df, hparams, label_to_index)
    train_module = HandwritingRecogTrainModule(hparams)
    train_model(train_module, data_module)


if __name__ == '__main__':
    test_handwriting_recognition()
