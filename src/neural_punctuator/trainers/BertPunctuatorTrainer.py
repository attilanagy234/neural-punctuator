import logging
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from neural_punctuator.base.BaseTrainer import BaseTrainer
from neural_punctuator.data.dataloader import BertDataset, collate, get_data_loaders, get_datasets
from neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW  # TODO
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from neural_punctuator.utils.data import get_target_weights
from neural_punctuator.utils.tensorboard import print_metrics


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


class BertPunctuatorTrainer(BaseTrainer):
    def __init__(self, model, preprocessor, config):
        super().__init__(model, preprocessor, config)

        if self._config.trainer.use_gpu:
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)

        else:
            self.device = torch.device('cpu')

        self.train_dataset, self.valid_dataset = get_datasets(config)
        self.train_loader, self.valid_loader = get_data_loaders(config)
        self.model = BertPunctuator(self._config.model.num_classes).to(self.device)

        if self._config.trainer.loss == 'NLLLoss':
            target_weights = torch.Tensor(get_target_weights(self.train_dataset.targets,
                                                             self._config.model.num_classes)).to(self.device)
            self.criterion = nn.NLLLoss(weight=target_weights)
        else:
            log.error('Please provide a proper loss function')
            exit(1)

        if self._config.model.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.model.learning_rate)

        elif self._config.model.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._config.model.learning_rate)
        else:
            log.error('Please provide a proper optimizer')
            exit(1)

        self.summary_writer = SummaryWriter(comment=self._config.experiment.name)

    def train(self):
        printer_counter = 0

        for epoch_num in range(self._config.trainer.num_epochs):
            log.info(f"Epoch #{epoch_num}")
            # Train loop
            for data in tqdm(self.train_loader):
                self.model.classifier.train()
                self.optimizer.zero_grad()

                text, targets = data
                output = self.model(text.to(self.device))

                loss = self.criterion(output.view(-1, self._config.model.num_classes), targets.to(self.device).view(-1))

                loss.backward()
                self.optimizer.step()

                loss = loss.item()

                if printer_counter != 0 and printer_counter % 10 == 0:
                    print_metrics(printer_counter, loss, self.summary_writer, 'train', model_name=self._config.experiment.name)
                printer_counter += 1

            # Valid loop
            for data in tqdm(self.valid_loader):
                self.model.eval()

                text, targets = data
                output = self.model(text.to(self.device))
                loss = self.criterion(output.view(-1, self._config.model.num_classes), targets.to(self.device).view(-1))
                loss = loss.item()

                print_metrics(printer_counter, loss, self.summary_writer, 'valid', model_name=self._config.experiment.name)



