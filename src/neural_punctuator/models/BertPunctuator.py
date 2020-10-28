import torch
from torch import nn
import torch.nn.functional as F


from neural_punctuator.base.BaseModel import BaseModel


class BertPunctuator(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = torch.hub.load(self._config.model.bert_github_repo, 'model', self._config.model.bert_variant_to_load)
        self.bert.eval()

        self.classifier = Classifier(self._config)

    def forward(self, x):
        embedding, _ = self.bert(x)
        output = self.classifier(embedding)
        output = F.log_softmax(output, dim=-1)
        return output


class Classifier(BaseModel):
    def __init__(self, config):
        super().__init__(None)
        self.linear1 = nn.Linear(config.model.bert_output_dim, config.model.linear_hidden_dim)
        self.linear2 = nn.Linear(config.model.linear_hidden_dim, config.model.num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x