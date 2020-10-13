# Papers
---
### Punctuation Prediction Model for Conversational Speech
 - Dataset: Fischer corpus (dialogues)
 - Used GloVe embeddings, trained on Common Web Crawl data
 - Models: CNN and BiLSTM
 - The final layer in both CNN and BLSTM model is fullyconnected and followed by a softmax activation - this layer is
applied separately at each time step to retrieve punctuation prediction for a given word.
 - A dialogue is represented as an ordered set of words, where words have several properties:
    - texutal representation
    - binary feature, represention which conversation side uttered the word.
    - a real number describing time offset (in seconds) at which the word started
    - a real number describing the duration (in seconds) of the word
    - a punctuation symbol, which appears after the word
