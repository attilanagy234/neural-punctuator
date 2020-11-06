import torch
import re
import logging
import sys
import yaml
import pandas as pd
import numpy as np
from dotmap import DotMap
from nltk.corpus import stopwords

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.addHandler(handler)

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

# TODO: We might have some unused stuff here, remove them!!

def get_mask_from_lengths(lengths):
    """
    Used for BERT masking
    """
    max_len = lengths.max()
    ids = torch.arange(max_len, 0, step=-1, device=lengths.device).long()
    return ids >= lengths.unsqueeze(1)


def get_weights_for_sampler(train_data, target_key):
    """
    Useful for torch's WeightedRandomSampler in case of class imbalance
    Params:
        train_data (Pandas DataFrame)
        target_key: (String): A header in train_data indicating the values to predict

    Returns:
        Weights for every train sample based on class frequency
    """

    # Convert categorical values to numbers
    train_data['factorized_target_key'], _ = pd.factorize(train_data[target_key])

    class_sample_count = np.array(
        [len(np.where(train_data['factorized_target_key'] == t)[0])
         for t in np.unique(train_data['factorized_target_key'])])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_data['factorized_target_key']])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    return samples_weight


def clean_text(text, lang):
    """
        text: a string
        return: modified initial string
    """
    _stopwords = get_stop_words(lang)
    text = text.lower()
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in _stopwords)  # remove stopwords from text
    return text


def get_stop_words(lang):
    try:
        return set(stopwords.words(lang))
    except Exception as e:
        log.error(e)
        raise ValueError


def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.FullLoader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config


def get_target_weights(targets, output_dim):
    targets = np.array(targets)
    weights = np.zeros((output_dim,))
    for t in range(output_dim):
        count = (targets == t).sum()
        weights[t] = count

    weights[0] *= 0.05
    weights /= weights.sum()
    return weights
