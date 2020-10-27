import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class BertDataset(Dataset):
    def __init__(self, data_path, prefix):

        self.seq_len = 512

        with open(data_path + prefix + "_data.pkl", 'rb') as f:
            texts, targets = pickle.load(f)
            self.encoded_texts = [word for t in texts for word in t]
            self.targets = [t for ts in targets for t in ts]

    def __getitem__(self, idx):
        return torch.LongTensor(self.encoded_texts[idx * self.seq_len: (idx+1) * self.seq_len]),\
               torch.LongTensor(self.targets[idx * self.seq_len: (idx+1) * self.seq_len])

    def __len__(self):
        return len(self.encoded_texts)//self.seq_len - 1


def collate(batch):
    texts, targets = zip(*batch)
    return torch.stack(texts), torch.stack(targets)
    # return pad_sequence(batch, batch_first=True, padding_value=PAD_ID)


def get_datasets(config):
    train_dataset = BertDataset(config.data.data_path, prefix="train")
    valid_dataset = BertDataset(config.data.data_path, prefix="valid")
    return train_dataset, valid_dataset


def get_data_loaders(train_dataset, valid_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.trainer.batch_size, num_workers=0, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.trainer.batch_size, collate_fn=collate)
    return train_loader, valid_loader
