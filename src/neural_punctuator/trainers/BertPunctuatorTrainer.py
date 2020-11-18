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
from neural_punctuator.utils.io import save, load
from neural_punctuator.utils.loss import WeightedBinaryCrossEntropy
from neural_punctuator.utils.metrics import get_total_grad_norm, get_eval_metrics
from neural_punctuator.utils.tensorboard import print_metrics
from neural_punctuator.utils.scheduler import LinearScheduler
import numpy as np

torch.manual_seed(69)
np.random.seed(69)
torch.backends.cudnn.deterministic = True


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


class BertPunctuatorTrainer(BaseTrainer):
    def __init__(self, model, preprocessor, config):
        super().__init__(model, preprocessor, config)

        if self._config.trainer.use_gpu:
            self.device = torch.device(self._config.trainer.use_gpu)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        self.train_dataset, self.valid_dataset = get_datasets(config)
        self.train_loader, self.valid_loader = get_data_loaders(self.train_dataset, self.valid_dataset, config)
        self.model = model.to(self.device)
        self.model.train()

        if self._config.trainer.loss == 'NLLLoss':
            target_weights = torch.Tensor(get_target_weights(self.train_dataset.targets,
                                                             self._config.model.num_classes)).clamp_max(1).to(self.device)
            self.criterion = nn.NLLLoss(weight=target_weights, reduction='none')
        else:
            log.error('Please provide a proper loss function')
            exit(1)

        optimizer_args = [
                {'params': self.model.base.parameters(), 'lr': self._config.trainer.base_learning_rate},
                {'params': self.model.classifier.parameters(), 'lr': self._config.trainer.classifier_learning_rate}
            ]
        if self._config.trainer.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(optimizer_args)

        elif self._config.trainer.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(optimizer_args)
        else:
            log.error('Please provide a proper optimizer')
            exit(1)

        if self._config.trainer.load_model:
            load(self.model, self.optimizer, self._config)

        # TODO: add to config
        self.sched = LinearScheduler(self.optimizer, self._config.trainer.warmup_steps)

        # TODO:
        self.all_valid_target = np.concatenate([targets.numpy() for _, targets in self.valid_loader])
        self.all_valid_target = self.all_valid_target[self.all_valid_target != -1]

        if self._config.debug.summary_writer:
            self.summary_writer = SummaryWriter(comment=self._config.experiment.name)
            #TODO: self.summary_writer.add_hparams(self._config.toDict(), {})
        else:
            self.summary_writer = None

    def train(self):
        printer_counter = 0

        for epoch_num in range(self._config.trainer.num_epochs):
            log.info(f"Epoch #{epoch_num}")

            # Train loop
            self.model.train()
            pbar = tqdm(self.train_loader)
            for data in pbar:
                self.optimizer.zero_grad()

                text, targets = data
                preds, binary_preds = self.model(text.to(self.device))

                # preds = preds[:, self._config.trainer.clip_seq: -self._config.trainer.clip_seq, :]
                # targets = targets[:, self._config.trainer.clip_seq:-self._config.trainer.clip_seq]

                # Mask some "empty" targets
                mask = ((targets == 0) & (np.random.rand(*targets.shape) < .1)) | (targets > 0)
                mask = mask.to(self.device)

                # Do not predict output after tokens which are not the end of a word
                not_a_word_mask = (targets == -1).to(self.device)
                word_mask = ~not_a_word_mask
                targets[not_a_word_mask] = 0

                losses = self.criterion(preds.reshape(-1, self._config.model.num_classes),
                                   targets.to(self.device).reshape(-1))
                mask = word_mask * mask
                # losses = mask.view(-1).to(self.device) * losses
                # loss = losses.sum() / mask.sum()
                loss = losses.mean()
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self._config.trainer.grad_clip)

                self.optimizer.step()
                self.sched.step()

                loss = loss.item()

                grads = get_total_grad_norm(self.model.parameters())
                pbar.set_postfix({"loss": loss, "grads": grads})

                print_metrics(printer_counter,
                              {"loss": loss, "grads": grads},
                              self.summary_writer, 'train',
                              model_name=self._config.model.name)
                printer_counter += 1

                if self._config.debug.break_train_loop:
                    break

            # Valid loop
            self.model.eval()
            valid_loss = 0
            all_valid_preds = []
            for data in tqdm(self.valid_loader):
                text, targets = data
                with torch.no_grad():
                    preds, _ = self.model(text.to(self.device))

                word_mask = targets != -1
                preds = preds[word_mask]
                targets = targets[word_mask]

                loss = self.criterion(preds.view(-1, self._config.model.num_classes), targets.to(self.device).view(-1))
                valid_loss += loss.mean().item()

                all_valid_preds.append(preds.detach().cpu().numpy())

            valid_loss /= len(self.valid_loader)
            all_valid_preds = np.concatenate(all_valid_preds)

            metrics = get_eval_metrics(self.all_valid_target, all_valid_preds, self._config)
            metrics["loss"] = valid_loss

            print_metrics(printer_counter, metrics, self.summary_writer, 'valid',
                          model_name=self._config.model.name)

            # Save model every epoch
            save(self.model, self.optimizer, epoch_num+1, metrics, self._config)

