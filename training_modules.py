import torch
import torch.nn as nn
import pytorch_lightning as pl

from modelling import HandwritingRecognition


class HandwritingRecogTrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.lr = hparams['lr']
        self.model = HandwritingRecognition(hparams['gru_input_size'], hparams['gru_hidden_size'],
                                            hparams['gru_num_layers'], hparams['num_classes'])
        self.criterion = nn.CTCLoss(blank=28)

    def forward(self, x):
        pass

    def intermediate_operation(self, batch):
        transformed_images = batch['transformed_images']
        labels = batch['labels']
        target_lens = batch['target_lens']

        output = self.model(transformed_images)

        N = output.size(1)
        input_length = output.size(0)
        input_lengths = torch.full(size=(N,), fill_value=input_length, dtype=torch.int32)

        loss = self.criterion(output, labels, input_lengths, target_lens)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.intermediate_operation(batch)

        self.log_dict({'train-loss': loss}, prog_bar=True, on_epoch=True,
                      on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.intermediate_operation(batch)

        self.log_dict({'val-loss': loss}, prog_bar=False, on_epoch=True,
                      on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
