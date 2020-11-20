import os
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from neural_punctuator.utils.data import get_config_from_yaml
from neural_punctuator.models.BertPunctuator import BertPunctuator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from neural_punctuator.data.dataloader import collate, get_data_loaders, get_datasets
from neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW
from torch import nn

from neural_punctuator.utils.io import save, load
from neural_punctuator.utils.metrics import get_total_grad_norm, get_eval_metrics
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
from itertools import product


def load_scores(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint['metrics']


def get_strict_f_score(report):
    return sum(float(report['cls_report'][output]['f1-score']) for output in ('period', 'question', 'comma')) / 3


def best_epoch_by_f_score(metrics):
    best_score = metrics[0]['strict_f_score']
    best_epoch = 0
    for i, m in enumerate(metrics):
        if m['strict_f_score'] > best_score:
            best_score = m['strict_f_score']
            best_epoch = i
    return best_epoch, best_score


def best_epoch_by_loss(metrics):
    best_loss = metrics[0]['loss']
    best_epoch = 0
    for i, m in enumerate(metrics):
        if m['loss'] < best_loss:
            best_loss = m['loss']
            best_epoch = i
    return best_epoch, best_loss


def combine(pred_num, all_valid_preds):
    relevant_preds = all_valid_preds[::pred_num]

    ps = []

    for i in range(relevant_preds.shape[0]):  # +512//pred_num-1):
        #     ps.append(relevant_preds[i, :pred_num])

        start_idx = max(0, i - 512 // pred_num + 1)
        end_idx = min(relevant_preds.shape[0], i + 1)

        p = []
        for j, k in enumerate(range(start_idx, end_idx)):
            j = end_idx - start_idx - j - 1
            #         print(k, j, relevant_preds[k][j*pred_num:(j+1)*pred_num].mean())
            p.append(relevant_preds[k][j * pred_num:(j + 1) * pred_num])
        #     print()
        p = np.stack(p)

        if p.shape[0] > 2:
            p = p[1:-1, :, :]

        ps.append(np.log(np.exp(p).mean(0)))

    ps = np.concatenate(ps)

    return ps


def combine(pred_num, preds):
    step_num = 512 // pred_num
    multi_preds = [preds[i::pred_num].reshape(-1, preds.shape[-1]) for i in range(pred_num)]
    for i in range(pred_num):
        start_idx = (pred_num - i - 1) * step_num
        end_idx = start_idx + (preds.shape[0] - (pred_num-1)*2) * step_num
        multi_preds[i] = multi_preds[i][start_idx:end_idx]

    multi_preds = np.stack(multi_preds)
    multi_preds = np.log(np.exp(multi_preds).mean(0))
    return multi_preds


class BertDataset(Dataset):
    def __init__(self, prefix, config, is_train=False):

        self.config = config
        self.is_train = is_train

        with open(self.config.data.data_path + prefix + "_data.pkl", 'rb') as f:
            texts, targets = pickle.load(f)
            self.encoded_texts = 512 * [0] + [word for t in texts for word in t] + 512 * [0]  # Add padding to both ends
            self.targets = 512 * [-1] + [t for ts in targets for t in ts] + 512 * [-1]

    def __getitem__(self, idx):
        if idx == 164:
            pass
        start_idx = (1+idx) * self.config.model.predict_step
        end_idx = start_idx + self.config.model.seq_len
        return torch.LongTensor(self.encoded_texts[start_idx: end_idx]),\
               torch.LongTensor(self.targets[start_idx: end_idx])

    def __len__(self):
        return int(np.ceil((len(self.encoded_texts)-1024)//self.config.model.predict_step))


def evaluate_multiple_predictions(model_name, model_type, predict_step, device):
    print(model_name, model_type)

    if model_type == 'by_f_score':
        epoch, _ = best_epoch_by_f_score(metrics[model_name])
    elif model_type == 'by_loss':
        epoch, _ = best_epoch_by_loss(metrics[model_name])
    else:
        raise ValueError("Model type not valid, options: by_f_score/by_loss")

    config = get_config_from_yaml(f'neural_punctuator/configs/config-{model_name}-unfreeze.yaml')
    config.trainer.load_model = f"{model_name}-epoch-{epoch + 1}.pth"

    config.model.predict_step = predict_step
    config.predict.batch_size = 128

    model = BertPunctuator(config)
    model.to(device)

    load(model, None, config)

    test_dataset = BertDataset("test", config)

    test_loader = DataLoader(test_dataset, batch_size=config.predict.batch_size, collate_fn=collate)

    model.eval()
    all_test_preds = []

    for data in tqdm(test_loader):
        text, targets = data
        with torch.no_grad():
            preds, _ = model(text.to(device))

        all_test_preds.append(preds.detach().cpu().numpy())

    all_test_target = test_dataset.targets[512:-512]
    all_test_preds = np.concatenate(all_test_preds)
    pred_num = config.model.seq_len // config.model.predict_step

    ps = combine(pred_num, all_test_preds)
    _targets = np.array(all_test_target[:ps.shape[0]])

    ps = ps[_targets != -1]
    _targets = _targets[_targets != -1]

    report = get_eval_metrics(_targets, ps, config)
    return report



if __name__ == "__main__":
    data_path = "/userhome/student/bial/neural-punctuator/models/"
    model_names = ["bert-base-uncased", "bert-base-cased", "albert-base-v1"]

    files = {}
    for model_name in model_names:
        f = sorted(glob(data_path + f"{model_name}-epoch*.*"), key=os.path.getmtime)
        files[model_name] = f

    # metrics = {}
    # for model_name in model_names:
    #     m = []
    #     for file in tqdm(files[model_name]):
    #         m.append(load_scores(file))
    #     metrics[model_name] = m

    # with open('metrics.pkl', 'wb') as f:
    #     pickle.dump(metrics, f)

    with open('metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    for _, m in metrics.items():
        for epoch in m:
            epoch['strict_f_score'] = get_strict_f_score(epoch)

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # for model_name, model_type in product(model_names, ('by_loss', 'by_f_score')):
    model_name = "albert-base-v1"
    model_type = "by_f_score"
    pred_num_for_token = 4
    predict_step = 512 // pred_num_for_token

    report = evaluate_multiple_predictions(model_name, model_type, predict_step, device)