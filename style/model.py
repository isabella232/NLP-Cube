# -*- coding: utf-8 -*-
import logging as log
from collections import OrderedDict
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertModel, get_constant_schedule_with_warmup

import pytorch_lightning as pl
from dataloader import sentiment_analysis_dataset
from test_tube import HyperOptArgumentParser
from utils import mask_fill


class StyleEstimator(pl.LightningModule):
    """
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(StyleEstimator, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # build model

        """
            Creates an Encoder model.
            Args:
                vocab_size (int): Number of classes/ Vocabulary size.
                emb_dim (int): Embeddings dimension.
                hidden_dim (int): LSTM hidden layer dimension.
                num_layers (int): Number of LSTM layers.
                lstm_dropout (float): LSTM dropout.
                dropout (float): Embeddings dropout.
                device : The device to run the model on.
        """

        super().__init__()

        self.vocab_size = 10000
        self.emb_dim = 300
        self.hidden_dim = 512
        self.num_layers = 3

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(self.emb_dim, int(self.hidden_dim / 2), self.num_layers, dropout=0.2, bidirectional=True,
                            batch_first=True)

        # Loss criterion initialization.
        self._loss = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, tokens, lengths):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        """
                Args:
                    input (tensor): The input of the encoder. It must be a 2-D tensor of integers. 
                        Shape: [batch_size, seq_len_enc].
                Returns:
                    A tuple containing the output and the states of the last LSTM layer. The states of the LSTM layer is also a
                    tuple that contains the hidden and the cell state, respectively . 
                        Output shape:            [batch_size, seq_len_enc, hidden_dim * 2]
                        Hidden/cell state shape: [num_layers*2, batch_size, hidden_dim]
                """

        X, X_lengths = input_tuple[0], input_tuple[1]

        # Creates the embeddings and adds dropout. [batch_size, seq_len] -> [batch_size, seq_len, emb_dim].
        embeddings = self.dropout(self.embedding(X))

        # pack padded sequences
        pack_padded_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(embeddings, X_lengths, batch_first=True)

        # now run through LSTM
        pack_padded_lstm_output, states = self.lstm(pack_padded_lstm_input)

        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_lstm_output, batch_first=True)

        return {'output': output, 'states': states}

        return {"logits": self.classification_head(sentemb)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
        return inputs, targets

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return sentiment_analysis_dataset(self.hparams, train, val, test)


    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)[0]
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )


    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)[0]
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )


    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)[0]
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--class_weights",
            default="ignore",
            type=str,
            help="Weights for each of the classes we want to tag.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=sys.maxsize,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
        )
        parser.add_argument(
            "--warmup_steps", default=200, type=int, help="Scheduler warmup steps.",
        )
        parser.opt_list(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout to be applied to the BERT embeddings.",
            tunable=True,
            options=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        # Data Args:
        parser.add_argument(
            "--train_csv",
            default="data/imdb_reviews_train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/imdb_reviews_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/imdb_reviews_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
