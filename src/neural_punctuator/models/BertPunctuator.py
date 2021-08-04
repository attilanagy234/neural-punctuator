from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


from neural_punctuator.base.BaseModel import BaseModel
from transformers import AutoTokenizer, AutoModel


class BertPunctuator(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.base = torch.hub.load(self._config.model.bert_github_repo, 'model', self._config.model.bert_variant_to_load)
        self.base = AutoModel.from_pretrained(self._config.model.load_model_repo, return_dict=False)

        if not self._config.trainer.train_bert:
            for param in self.base.parameters():
                param.requires_grad = False

        self.classifier = Classifier(self._config)

    def forward(self, x):
        if self._config.trainer.train_bert:
            embedding, _ = self.base(x)
        else:
            with torch.no_grad():
                embedding, _ = self.base(x)

        output, binary_output = self.classifier(embedding)
        output = F.log_softmax(output, dim=-1)
        return output, binary_output

    def train(self, mode=True):
        if mode:
            if self._config.trainer.train_bert:
                self.base.train()
            else:
                self.base.eval()
            self.classifier.train()
        else:
            self.base.eval()
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
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(config.model.dropout)
        self.linear2 = nn.Linear(config.model.linear_hidden_dim, config.model.num_classes)
        self.binary_classifier = nn.Linear(config.model.linear_hidden_dim, 1)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.activation(self.linear1(x))
        x = self.dropout2(x)
        binary_output = torch.sigmoid(self.binary_classifier(x))
        x = self.linear2(x)
        return x, binary_output
