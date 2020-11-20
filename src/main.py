from neural_punctuator.utils.data import get_config_from_yaml
from neural_punctuator.wrappers.BertPunctuatorWrapper import BertPunctuatorWrapper

if __name__ == '__main__':
    config = get_config_from_yaml(
        'neural_punctuator/configs/config-szeged-bert-base-multilingual-uncased-unfreeze.yaml')
    pipe = BertPunctuatorWrapper(config)
    pipe.train()

