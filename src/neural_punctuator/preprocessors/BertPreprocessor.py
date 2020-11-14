import os
import json
import torch
import pickle
from tqdm import tqdm
from neural_punctuator.base.BasePreprocessor import BasePreprocessor
from transformers import AutoTokenizer, AutoModel


class BertPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config)
        self._tokenizer = torch.hub.load(self._config.model.bert_github_repo, 'tokenizer', self._config.model.bert_variant_to_load)

    # def transform(self, input_texts):
    #     encoded_texts = []
    #     for text in tqdm(input_texts):
    #         encoded = self.encode_text(text)
    #         encoded_texts.append(encoded)
    #
    #     return encoded_texts
    #
    # def inverse_transform(self, *args):
    #     # TODO: should be implemented once we want to sanity check on textual outputs of the model
    #     raise NotImplementedError
    #
    # def encode_text(self, text):
    #     for ew in escape_words:
    #         text = text.replace(ew, '')
    #
    #     text = text.replace('!', '.')
    #     text = text.replace(';', '.')
    #     text = text.replace(':', ',')
    #     text = text.replace('--', ',')
    #     text = text.replace('-', ',')
    #
    #     return self._tokenizer.encode(text)
    #
    # def dump_encoded_data_to_pickle(self, data):
    #     filename = f'{self._config.data_path}_encoded.pkl'
    #     with open(filename, 'wb') as f:
    #         pickle.dump(data, f)


if __name__ == "__main__":
    # TODO: remove these stuff, entrypoint should not be here
    data_path = os.environ['DATA_PATH']
    file_path = data_path + "ted_talks-25-Apr-2012.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
