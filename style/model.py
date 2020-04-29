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
from test_tube import HyperOptArgumentParser

from .utils import mask_fill
from .lookup import Lookup
from .dataloader import MyDataset

class StyleEstimator(pl.LightningModule):
    """
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(StyleEstimator, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        #self.lookup = Lookup(type="bpe")
        #self.lookup.load(file_prefix="lookup/bpe/tok")

        # build model

        self.vocab_size = 10000
        self.emb_dim = 300
        self.hidden_dim = 512
        self.num_layers = 3
        self.output_dim = 10

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.sentence_lstm = nn.LSTM(self.emb_dim, int(self.hidden_dim / 2), self.num_layers, dropout=0.2, bidirectional=True,
                            batch_first=True)
        self.instance_lstm = nn.LSTM(self.hidden_dim, int(self.hidden_dim / 2), self.num_layers, dropout=0.2,
                                     bidirectional=True,
                                     batch_first=True)
        self.instance_mlp = nn.Linear(self.hidden_dim, self.output_dim)

        # Loss criterion initialization.
        self._loss = nn.MSELoss()#reduction="none"

    def forward(self, x_tensor, x_lengths):
        #batch_output = torch.zeros(len(inputs), self.output_dim) # [batch_size, output_size]
        batch_output = []
        # x_tensor is a [batch_size, no_sentences, max_len]
        # x_lengths is a [batch_size * no_sentences]
        #torch.save(x_tensor, "x.x")
        #torch.save(x_lengths, "x.len")

        batch_size = x_tensor.size(0)
        no_sentences = x_tensor.size(1)
        max_len = x_tensor.size(2)

        # process sentences together
        #print(x_tensor.size())
        x = x_tensor.view(-1, max_len) # [batch_size * no_sentences, max_len]
        embeddings = self.dropout(self.embedding(x))  # [batch_size * no_sentences, max_len, emb_dim]

        # pack padded sequences
        pack_padded_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(embeddings, x_lengths, batch_first=True,
                                                                         enforce_sorted=False)

        # now run through LSTM
        pack_padded_lstm_output, _ = self.sentence_lstm(pack_padded_lstm_input)

        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_lstm_output, batch_first=True)

        # print(output.size()) # [batch_size * no_sentences, max_len, hidden_dim]

        # create batch for instance lstm
        last_states = output[:, -1, :] # [batch_size * no_sentences, hidden_dim]
        instance_batch = last_states.view(batch_size, no_sentences, self.hidden_dim) # [batch_size, no_sentences, hidden_dim]

        # run instance lstm
        outputs, _ = self.instance_lstm(instance_batch)  # [batch_size, no_sentences, hidden_dim]

        instance_encoding = outputs[:, -1, :]  # [batch_size, hidden_dim]
        batch_output = self.instance_mlp(instance_encoding.squeeze())  # [batch_size, output_size]

        #input("asdA")


        """ 
        sentence_hidden_states = []
        for instance_index, instance in enumerate(inputs): # inputs is a list of batch_size elements of strings
            # run individual sentences
            no_sentences = instance.size(0)
            max_len = instance.size(1)
            embeddings = self.dropout(self.embedding(instance)) # [no_sentences, max_len, emb_dim]
            outputs, _ = self.sentence_lstm(embeddings) # output is [no_sentences, max_len, hidden_dim] , i.e. (batch, seq_len, num_directions * hidden_size)
            last_hidden_states = outputs[:, -1, :].unsqueeze(0) # [1, no_sentences, hidden_dim]
            sentence_hidden_states.append(last_hidden_states)
        sentence_hidden_states = torch.cat(sentence_hidden_states, dim=0) # [batch_size, no_sentences, hidden_dim]

        outputs, _ = self.instance_lstm(sentence_hidden_states)  # [batch_size, no_sentences, hidden_dim]

        instance_encoding = outputs[:, -1, :]  # [batch_size, hidden_dim]
        batch_output = self.instance_mlp(instance_encoding.squeeze())  # [batch_size, output_size]

        """
        #print(batch_output.size())
        #input("Asdasd")
        """
        # pack padded sequences
        pack_padded_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(embeddings, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        pack_padded_lstm_output, states = self.lstm(pack_padded_lstm_input)
        
        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_lstm_output, batch_first=True)   
        """

        """ 
            #states = torch.zeros(1, len(instance), self.hidden_dim)
            states = []
            for i, sentence in enumerate(instance): # sentence is a string (a sentence)
                # Creates the embeddings and adds dropout. [1, seq_len] -> [1, seq_len, emb_dim].
                tsentence = torch.tensor([sentence]) #[1, seq_len]
                embeddings = self.dropout(self.embedding(tsentence))

                # [1, seq_len, emb_dim] -> [1, 1, 512]
                outputs, _ = self.sentence_lstm(embeddings) # output is [1,2,512] -> (
                last_hidden = outputs[:, -1, :] # [batch_size=1, hidden]
                #states[:, i, :] = last_hidden
                states.append(last_hidden.unsqueeze(1))

            # run through instance lstm
            states = torch.cat(states, dim=1)

            outputs, _ = self.instance_lstm(states) # [1, 9, 512]
            #print(outputs.size())
            instance_encoding = outputs[:,-1,:] # [batch_size=1, hidden]
            output_mlp = self.instance_mlp(instance_encoding.squeeze()) # [output_size]
            #batch_output[instance_index,:] = output_mlp
            batch_output.append(output_mlp.unsqueeze(0))
        batch_output = torch.cat(batch_output, dim=0)
        # print(batch_output.size())
        """
        return batch_output

    def loss(self, prediction, target) -> torch.tensor:
        # prediction is a [batch_size, output_size]
        # target is a [batch_size, output_size]

        lss = self._loss(prediction, target)
        #print(lss.size())
        return lss

    def prepare_sample(self, inputs: list) -> (list, list):
        """print("prepare sample")
        for x in inputs:
            print(x)
        print("_"*10)
        """
        #inputs is a list of tuples (x,y) with batch_size elements

        x = [x[0] for x in inputs]
        y = [x[1] for x in inputs]

        batch_size = len(inputs)
        no_of_sentences = len(x[0])

        # get max lengths
        max_len = 0
        for instance in x:
            max_len = max([max_len] + [len(sentence) for sentence in instance])

        x_tensor = [] # should be [batch_size, no_of_sentences, max_len]
        x_lengths = [] # tensor of [batch_size * no_of_sentences]
        for instance in x:
            instance_tensor = []
            for sentence in instance:
                sentence_len = len(sentence)
                x_lengths.append(sentence_len)
                padded_sentence = torch.tensor(sentence + [0] * (max_len - sentence_len), dtype=torch.long)
                instance_tensor.append(padded_sentence.unsqueeze(0)) # lista cu [1, max_len]
            instance_tensor = torch.cat(instance_tensor, dim=0) # [no_of_sentences, max_len]
            x_tensor.append(instance_tensor.unsqueeze(0)) # list of [1, no_of_sentences, max_len]
        x_tensor = torch.cat(x_tensor, dim=0) # [batch_size, no_of_sentences, max_len]

        #print(x_tensor.size())

        y_tensor = torch.tensor(y, dtype=torch.float)
        return x_tensor, x_lengths, y_tensor

        #input("Asd")

        """ 
        # x is a list of batch_size arrays of ints
        # get max_size of elements and pad, for each x instance
        x_tensor = []
        for instance in x:
            max_size = max([len(sentence) for sentence in instance])
            no_of_sentences = len(instance)
            # target is to create a [#_of_sentences, max_size] tensor padded with zero
            instance_tensor = []
            for sentence in instance:
                padded_sentence = torch.tensor(sentence + [0]*(max_size-len(sentence)), dtype=torch.long)
               
                instance_tensor.append(padded_sentence.unsqueeze(0))
            instance_tensor = torch.cat(instance_tensor, dim=0) # [no_of_sentences, max_size?? cum merg efectiv prin lstm??
            x_tensor.append(instance_tensor)

        y_tensor = torch.tensor(y, dtype=torch.float)
        return x_tensor, y_tensor
        """

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        x_tensor, x_lengths, y_tensor = batch

        model_out = self.forward(x_tensor, x_lengths)
        loss_val = self.loss(model_out, y_tensor)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        x_tensor, x_lengths, y_tensor = batch

        model_out = self.forward(x_tensor, x_lengths)
        loss_val = self.loss(model_out, y_tensor)

        output = OrderedDict({"val_loss": loss_val})

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
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        pass

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = MyDataset(folder_path="data/gsts", train=True)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )


    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = MyDataset(folder_path="data/gsts", valid=True)
        #print(self._dev_dataset.xs[0])
        #print(self._dev_dataset.ys[0])
        return DataLoader(
            dataset=self._dev_dataset,
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
