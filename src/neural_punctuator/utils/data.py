import torch
import re
import logging
import sys
import yaml
import pandas as pd
import numpy as np
from dotmap import DotMap
from nltk.corpus import stopwords
from sklearn.utils.class_weight import compute_class_weight


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.addHandler(handler)

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

# TODO: We might have some unused stuff here, remove them!!


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


def get_target_weights(targets, output_dim, reduce_empty=True):
    import warnings
    warnings.filterwarnings("ignore")

    weights = compute_class_weight('balanced', range(-1, 4), targets)[1:] # exclude -1
    # targets = np.array(targets)
    # weights = np.zeros((output_dim,))
    # for t in range(output_dim):
    #     count = (targets == t).sum()
    #     weights[t] = count
    #
    # if reduce_empty:
    #     weights[0] *= 0.05
    #
    # weights /= weights.sum()

    return weights
