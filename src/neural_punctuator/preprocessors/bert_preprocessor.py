import os
import json
import torch
import pickle
from tqdm import tqdm


escape_words = (' (Laughter) ', ' (Applause) ')


def encode_text(text, tokenizer):
    for ew in escape_words:
        text = text.replace(ew, '')

    text = text.replace('!', '.')
    text = text.replace(';', '.')
    text = text.replace(':', ',')
    text = text.replace('--', ',')
    text = text.replace('-', ',')

    encoded_text = albert_tokenizer.encode(text)

    return encoded_text


if __name__ == "__main__":

    data_path = os.environ['DATA_PATH']
    file_path = data_path + "ted_talks-25-Apr-2012.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcripts = [d['transcript'] for d in data]

    albert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'albert-base-v1')

    encoded_texts = []
    for text in tqdm(transcripts):
        encoded = encode_text(text, albert_tokenizer)
        encoded_texts.append(encoded)

    with open(data_path + "encoded.pkl", 'wb') as f:
        pickle.dump(encoded_texts, f)
