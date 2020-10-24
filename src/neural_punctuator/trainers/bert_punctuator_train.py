import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from neural_punctuator.data.dataloader import BertDataset, collate
from neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW  # TODO
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from neural_punctuator.utils.data import get_target_weights
from neural_punctuator.utils.metrics import get_eval_metrics, get_total_grad_norm
from neural_punctuator.utils.scheduler import LinearScheduler
from neural_punctuator.utils.tensorboard import print_metrics

import warnings
warnings.filterwarnings('ignore')


output_dim = 4
batch_size = 4
data_path = "D:/Downloads/ted_dataset.tar/ted_dataset/"
lr = 1e-4
device = torch.device("cuda:0")
torch.cuda.set_device(device)
model_name = "albert"
epochs = 20
grad_clip = 10
warmup_steps = 200
clip_seq = 32


if __name__ == '__main__':
    train_dataset = BertDataset(data_path, prefix="train")
    valid_dataset = BertDataset(data_path, prefix="valid")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)

    all_valid_target = np.concatenate([targets.numpy()[:, clip_seq:-clip_seq] for _, targets in valid_loader])

    model = BertPunctuator(output_dim).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    sched = LinearScheduler(optimizer, warmup_steps)

    target_weights = torch.Tensor(get_target_weights(train_dataset.targets, output_dim)).to(device)
    criterion = nn.NLLLoss(weight=target_weights, reduction='none')

    summary_writer = SummaryWriter(comment=model_name)
    printer_counter = 0

    for epoch_num in range(epochs):
        print(f"Epoch #{epoch_num}")

        model.classifier.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()

            text, targets = data
            preds = model(text.to(device))

            preds = preds[:, clip_seq: -clip_seq, :]
            targets = targets[:, clip_seq:-clip_seq]
            losses = criterion(preds.reshape(-1, output_dim),
                               targets.to(device).reshape(-1))
            mask = ((targets == 0) & (np.random.rand(*targets.shape) < .05)) | (targets != 0)
            losses = mask.view(-1).to(device) * losses
            loss = losses.sum() / mask.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            sched.step()

            loss = loss.item()

            if printer_counter != 0 and printer_counter % 10 == 0:
                grads = get_total_grad_norm(model.parameters())
                print_metrics(printer_counter, {"loss": loss, "grads": grads}, summary_writer, 'train', model_name=model_name)
            printer_counter += 1


        # Validate
        model.eval()
        valid_loss = 0

        all_valid_preds = []

        for data in tqdm(valid_loader):
            text, targets = data
            preds = model(text.to(device))
            # loss = criterion(preds.view(-1, output_dim), targets.to(device).view(-1))
            # valid_loss += loss.item()

            preds = preds[:, clip_seq: -clip_seq, :]
            targets = targets[:, clip_seq:-clip_seq]
            losses = criterion(preds.reshape(-1, output_dim),
                               targets.to(device).reshape(-1))
            mask = ((targets == 0) & (np.random.rand(*targets.shape) < .05)) | (targets != 0)
            losses = mask.view(-1).to(device) * losses
            loss = losses.sum() / mask.sum()

            all_valid_preds.append(preds.detach().cpu().numpy())

        valid_loss /= len(valid_loader)
        all_valid_preds = np.concatenate(all_valid_preds)

        metrics = get_eval_metrics(all_valid_target, all_valid_preds)
        metrics["loss"] = valid_loss

        print_metrics(printer_counter, metrics, summary_writer, 'valid', model_name=model_name)
