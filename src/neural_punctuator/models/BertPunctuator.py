import torch
from torch import nn
import torch.nn.functional as F


from neural_punctuator.base.BaseModel import BaseModel


class BertPunctuator(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = torch.hub.load(self._config.model.bert_github_repo, 'model', self._config.model.bert_variant_to_load)
        self.bert.eval()
        self.classifier = nn.Linear(self._config.model.bert_output_dim, self._config.model.num_classes)

    def forward(self, x):
        embedding, _ = self.bert(x)
        output = self.classifier(embedding)
        output = F.log_softmax(output, dim=-1)
        return output
