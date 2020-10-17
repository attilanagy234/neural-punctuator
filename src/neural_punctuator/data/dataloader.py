import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class BertDataset(Dataset):
    def __init__(self, file_path, batch_size):

        self.seq_len = 512
        self.batch_size = batch_size

        with open(file_path, 'rb') as f:
            encoded_texts = pickle.load(f)

    def __getitem__(self, idx):
        # TODO: not continous text
        return torch.LongTensor(self.encoded_texts[idx * self.seq_len : (idx+1) * self.seq_len])

    def __len__(self):
        len(self.encoded_texts)//self.seq_len - 1


def collate(batch, seq_len, PAD_ID):
    return pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
