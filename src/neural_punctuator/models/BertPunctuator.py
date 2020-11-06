from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


from neural_punctuator.base.BaseModel import BaseModel


class BertPunctuator(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = torch.hub.load(self._config.model.bert_github_repo, 'model', self._config.model.bert_variant_to_load)

        if not self._config.trainer.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = Classifier(self._config)

    def forward(self, x):
        if self._config.trainer.train_bert:
            embedding, _ = self.bert(x)
        else:
            with torch.no_grad():
                embedding, _ = self.bert(x)

        output = self.classifier(embedding)
        output = F.log_softmax(output, dim=-1)
        return output

    def train(self, mode=True):
        if mode:
            if self._config.trainer.train_bert:
                self.bert.train()
            else:
                self.bert.eval()
            self.classifier.train()
        else:
            self.bert.eval()
            self.classifier.eval()
        return self

    def eval(self):
        self.train(False)
        return self


class Classifier(BaseModel):
    def __init__(self, config):
        super().__init__(None)
        self.dropout1 = nn.Dropout(config.model.dropout)
        self.linear1 = nn.Linear(config.model.bert_output_dim, config.model.linear_hidden_dim)
        self.dropout2 = nn.Dropout(config.model.dropout)
        self.linear2 = nn.Linear(config.model.linear_hidden_dim, config.model.num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(x)
        x = self.activation(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        return x