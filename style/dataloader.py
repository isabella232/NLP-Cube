# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser
import os, ntpath, pickle, json, sys
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, train: bool = False, valid: bool = False, test: bool = False, folder_path: str = ""):
        if not os.path.exists(os.path.join("data", "train.x")):
            self._create_ds(folder_path = folder_path)

        # load the dataset
        self.xs = []
        self.ys = []

        if train:
            with open("data/train.x", "r", encoding="utf8") as f:
                self.xs = json.load(f)
            with open("data/train.y", "rb") as f:
                self.ys = pickle.load(f)
            self.xs = self.xs[:100]
            self.ys = self.ys[:100]
            print("we have {} training instances.".format(len(self.xs)))
            return
        if valid:
            with open("data/test.x", "r", encoding="utf8") as f:
                self.xs = json.load(f)
            with open("data/test.y", "rb") as f:
                self.ys = pickle.load(f)
            self.xs = self.xs[:50]
            self.ys = self.ys[:50]
            print("we have {} validation instances.".format(len(self.xs)))
            return


    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]




if __name__ == "__main__":
    obj = MyDataset(folder_path="data/gsts", train=True)
    print(obj)
