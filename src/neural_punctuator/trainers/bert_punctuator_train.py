import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_punctuator.data.dataloader import BertDataset, collate
from neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW  # TODO
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from neural_punctuator.utils.data import get_target_weights
from neural_punctuator.utils.tensorboard import print_metrics

output_dim = 4
batch_size = 8
data_path = "D:/Downloads/ted_dataset.tar/ted_dataset/"
lr = 1e-3
device = torch.device("cuda:0")
torch.cuda.set_device(device)
model_name = "Albert"
epochs = 20


if __name__ == '__main__':
    train_dataset = BertDataset(data_path, prefix="train")
    valid_dataset = BertDataset(data_path, prefix="valid")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)

    model = BertPunctuator(output_dim).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    target_weights = torch.Tensor(get_target_weights(train_dataset.targets, output_dim)).to(device)
    criterion = nn.NLLLoss(weight=target_weights)

    summary_writer = SummaryWriter(comment=model_name)
    printer_counter = 0

    for epoch_num in range(epochs):
        print(f"Epoch #{epoch_num}")
        for data in tqdm(train_loader):
            model.classifier.train()
            optimizer.zero_grad()

            text, targets = data
            output = model(text.to(device))

            loss = criterion(output.view(-1, output_dim), targets.to(device).view(-1))

            loss.backward()
            optimizer.step()

            loss = loss.item()

            if printer_counter != 0 and printer_counter % 10 == 0:
                print_metrics(printer_counter, loss, summary_writer, 'train', model_name=model_name)
            printer_counter += 1

        # Validate
        for data in tqdm(valid_loader):
            model.eval()

            text, targets = data
            output = model(text.to(device))
            loss = criterion(output.view(-1, output_dim), targets.to(device).view(-1))
            loss = loss.item()

            print_metrics(printer_counter, loss, summary_writer, 'valid', model_name=model_name)
