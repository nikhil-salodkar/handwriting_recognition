import os
import streamlit as st

from training_modules import HandwritingRecogTrainModule


@st.experimental_memo
def get_model_details():
    path = './lightning_logs/CNNR_run_64_2grulayers_0.3dropout/3182ng3f/checkpoints'
    model_weights = 'epoch=47-val-loss=0.190-val-exact-match=83.1511001586914-val-char-error-rate=0.042957037687301636.ckpt'
    model_path = os.path.join(path, model_weights)
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
    return model_path, hparams, label_to_index, index_to_labels


@st.experimental_memo
def load_trained_model(model_path):

    model = HandwritingRecogTrainModule.load_from_checkpoint(model_path)
    return model


def get_predictions(image):
    model_path, hparams, label_to_index, index_to_labels = get_model_details()
    model = load_trained_model(model_path)
    model_prediction = model(image)

    return model_prediction
