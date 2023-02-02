import os
from PIL import Image
import torch
import torch.nn as nn
from ctc_decoder import best_path, beam_search
import pytorch_lightning as pl
from torchmetrics import CharErrorRate
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from torchvision.utils import make_grid

from modelling import HandwritingRecognition


class HandwritingRecogTrainModule(pl.LightningModule):
    def __init__(self, hparams, index_to_labels, label_to_index):
        super().__init__()
        # save_hyperparameters saves the parameters in the signature in the form of dict
        self.save_hyperparameters()
        self.chars = ' -ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.model = HandwritingRecognition(self.hparams['hparams']['gru_input_size'], self.hparams['hparams']['gru_hidden_size'],
                                            self.hparams['hparams']['gru_num_layers'], self.hparams['hparams']['num_classes'])
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True, reduction='mean')
        self.transforms = Compose([Resize((self.hparams['hparams']['input_height'], self.hparams['hparams']['input_width'])), Grayscale(),
                                                              ToTensor()])
        self.char_metric = CharErrorRate()

    def forward(self, the_image):
        out = self.model(the_image)
        out = out.permute(1, 0, 2)
        out = torch.exp(out)
        return out

    def intermediate_operation(self, batch):
        transformed_images = batch['transformed_images']
        labels = batch['labels']
        target_lens = batch['target_lens']

        output = self.model(transformed_images)

        N = output.size(1)
        input_length = output.size(0)
        input_lengths = torch.full(size=(N,), fill_value=input_length, dtype=torch.int32)

        loss = self.criterion(output, labels, input_lengths, target_lens)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, preds = self.intermediate_operation(batch)
        with torch.inference_mode():
            preds = preds.permute(1, 0, 2)
            preds = torch.exp(preds)
            ground_truth = batch['labels']
            target_lens = batch['target_lens']
            ground_truth = ground_truth.cpu().detach().numpy()
            target_lens = target_lens.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            actual_predictions = []
            for pred in preds:
                actual_predictions.append(best_path(pred, self.chars))
            exact_matches = 0
            actual_ground_truths = []
            for i, predicted_string in enumerate(actual_predictions):
                ground_truth_sample = ground_truth[i][0:target_lens[i]]
                ground_truth_string = [self.hparams.index_to_labels[index] for index in ground_truth_sample]
                ground_truth_string = ''.join(ground_truth_string)
                actual_ground_truths.append(ground_truth_string)
                if predicted_string == ground_truth_string:
                    exact_matches += 1
            exact_match_percentage = (exact_matches / len(preds)) * 100
            char_error_rate = self.char_metric(actual_predictions, actual_ground_truths)
            self.log_dict({'train-loss': loss, 'train-exact-match': exact_match_percentage,
                           'train-char_error_rate': char_error_rate}, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.intermediate_operation(batch)
        preds = preds.permute(1, 0, 2)
        preds = torch.exp(preds)
        ground_truth = batch['labels']
        target_lens = batch['target_lens']
        ground_truth = ground_truth.cpu().detach().numpy()
        target_lens = target_lens.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        actual_predictions = []
        for pred in preds:
            actual_predictions.append(best_path(pred, self.chars))
        exact_matches = 0
        actual_ground_truths = []
        for i, predicted_string in enumerate(actual_predictions):
            ground_truth_sample = ground_truth[i][0:target_lens[i]]
            ground_truth_string = [self.hparams.index_to_labels[index] for index in ground_truth_sample]
            ground_truth_string = ''.join(ground_truth_string)
            actual_ground_truths.append(ground_truth_string)
            if predicted_string == ground_truth_string:
                exact_matches += 1
        char_error_rate = self.char_metric(actual_predictions, actual_ground_truths)
        exact_match_percentage = (exact_matches / len(preds)) * 100
        # visualizing predictions results for validation samples every epoch
        if batch_idx % self.trainer.num_val_batches[0] == 0:
            small_batch = batch['transformed_images'][0:16]
            small_batch_predictions = actual_predictions[0:16]
            captions = small_batch_predictions
            sampled_img_grid = make_grid(small_batch)
            self.logger.log_image('Sample_Images', [sampled_img_grid], caption=[str(captions)])

        self.log_dict({'val-loss': loss, 'val-exact-match': exact_match_percentage,
                       'val-char-error-rate': char_error_rate}, prog_bar=False, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['hparams']['lr'])


def convert_to_torchscript(trained_path):
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition', 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256, 'input_height': 36, 'input_width': 324, 'gru_hidden_size': 128,
        'gru_num_layers': 1, 'num_classes': 28
    }
    index_to_labels = {0: ' ', 1: '-', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E', 7: 'F', 8: 'G', 9: 'H', 10: 'I',
                       11: 'J', 12: 'K', 13: 'L', 14: 'M', 15: 'N', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S',
                       21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}
    label_to_index = {' ': 0, '-': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11,
                      'K': 12, 'L': 13, 'M': 14, 'N': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22,
                      'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27}
    model = HandwritingRecogTrainModule(hparams, index_to_labels, label_to_index)
    script = model.to_torchscript()
    print("The script:", script)
    torch.jit.save(script, './final-models/torchscript-model/handwritten-name_new.pt')

def test_model():
    pl.seed_everything(2564)
    hparams = {
        'train_img_path': './data/kaggle-handwriting-recognition/train_v2/train/',
        'lr': 1e-3, 'val_img_path': './data/kaggle-handwriting-recognition/validation_v2/validation/',
        'test_img_path': './data/kaggle-handwriting-recognition/test_v2/test/',
        'data_path': './data/kaggle-handwriting-recognition', 'gru_input_size': 256,
        'train_batch_size': 64, 'val_batch_size': 256, 'input_height': 36, 'input_width': 324, 'gru_hidden_size': 128,
        'gru_num_layers': 1, 'num_classes': 28
    }
    index_to_labels = {0: ' ', 1: '-', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E', 7: 'F', 8: 'G', 9: 'H', 10: 'I',
                       11: 'J', 12: 'K', 13: 'L', 14: 'M', 15: 'N', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S',
                       21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}
    label_to_index = {' ': 0, '-': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11,
                      'K': 12, 'L': 13, 'M': 14, 'N': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22,
                      'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27}

    model = HandwritingRecogTrainModule.load_from_checkpoint(
        './lightning_logs/CNNR_run_new_version/108xqa9y/checkpoints/'
        'epoch=21-val-loss=0.206-val-exact-match=81.46109771728516-val-char-error-rate=0.04727236181497574.ckpt')
    input_image = Image.open(os.path.join(hparams['train_img_path'], 'TRAIN_96628.jpg'))
    output = model(input_image)
    print(output)


def test_inference():
    transforms = Compose([Resize((36, 324)), Grayscale(), ToTensor()])
    input_image = Image.open(os.path.join('./data/kaggle-handwriting-recognition/train_v2/train/', 'TRAIN_96628.jpg'))
    transformed_image = transforms(input_image)
    # path = './lightning_logs/CNNR_run_64_2grulayers_0.3dropout/3182ng3f/checkpoints'
    # model_weights = 'epoch=47-val-loss=0.190-val-exact-match=83.1511001586914-val-char-error-rate=0.042957037687301636.ckpt'
    # trained_path = os.path.join(path, model_weights)
    # model = HandwritingRecogTrainModule.load_from_checkpoint(trained_path)
    # transformed_image = torch.unsqueeze(transformed_image, 0)
    # model.eval()
    # out = model(transformed_image)
    script_path = './final-models/torchscript-model/handwritten-name_new.pt'
    scripted_module = torch.jit.load(script_path)
    out = scripted_module(transformed_image)
    print("The final out shape:", out.shape)
    print("The final out is :", out)
    out = out.cpu().detach().numpy()
    chars = ' -ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for sample in out:
        predicted_string = beam_search(sample, chars, beam_width=2)
    print(predicted_string)


def test_convert_to_torchscript():
    path = './lightning_logs/CNNR_run_new_version/108xqa9y/checkpoints/'
    model_weights = 'epoch=21-val-loss=0.206-val-exact-match=81.46109771728516-val-char-error-rate=0.04727236181497574.ckpt'
    trained_path = os.path.join(path, model_weights)
    convert_to_torchscript(trained_path)


if __name__ == '__main__':
    test_inference()

