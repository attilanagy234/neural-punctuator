import torch
from torch import nn
import torch.nn.functional as F


from neural_punctuator.base.BaseModel import BaseModel


class BertPunctuator(BaseModel):
    def __init__(self, output_dim):
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'albert-base-v1')
        self.bert.eval()

        self.classifier = nn.Linear(768, output_dim)

    def forward(self, x):
        embedding = self.bert(x)
        output = self.classifier(embedding)
        output = F.log_softmax(output, dim=-1)
        return output
