def create_target(text, tokenizer, id2target):
    encoded_words, targets = [], []
    
    words = text.split(' ')

    for word in words:
        target = -1
        for target_token, target_id in target_token2id.items():
            if word.endswith(target_token):
                word = word.rstrip(target_token)
                target = id2target[target_id]

        encoded_word = tokenizer.encode(word, add_special_tokens=False)
        encoded_words = encoded_words + encoded_word
        targets = targets + [[-1] * (len(encoded_word)-1)][0] + [target]
#         print([tokenizer._convert_id_to_token(ew) for ew in encoded_word], target)

    return encoded_words, targets