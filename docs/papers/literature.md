# Papers
---
### Punctuation Prediction Model for Conversational Speech
https://arxiv.org/pdf/1807.00543.pdf
 - Dataset: Fischer corpus (dialogues)
 - Retains blanks (no punctuation), dots, commas and question marks. Other punctuation classes were rejected (converted to blanks) due to their low frequency (e.g. exclamation marks or triple dots) or the fact that it is modeled by other properties of the representation (double dash - that marks an interruptions).
 - Used GloVe embeddings (300 dim), trained on Common Web Crawl data
 - Models: CNN and BiLSTM
 - The final layer in both CNN and BLSTM model is fullyconnected and followed by a softmax activation - this layer is
applied separately at each time step to retrieve punctuation prediction for a given word.
 - A dialogue is represented as an ordered set of words, where words have several properties:
    - texutal representation
    - binary feature, represention which conversation side uttered the word.
    - a real number describing time offset (in seconds) at which the word started
    - a real number describing the duration (in seconds) of the word
    - a punctuation symbol, which appears after the word

### Punctuation Prediction for Unsegmented Transcript Based on Word Vector
http://www.lrec-conf.org/proceedings/lrec2016/pdf/103_Paper.pdf
- Treats the classification problem as whether a word in the sequence is followed by a punctuation mark.
- Classify whether a punctuation mark should be inserted after the third word of a 5-words sequence and which kind of punctuation mark the inserted one should be.
- Gives a nice literature overview --> we could refer to these papers in our own literature overview.
- 4 classes in total: O (means no punctuation mark followed), COMMA, PERIOD and QUESTION.
- Exclamation marks or semicolons are classified as PERIOD, while colons or dashes are classified as COMMA. There are no brackets in our data and all the other punctuation marks, such as quotation marks, are just ignored.
- Dataset: TED Talks (IWSLT 2012 Evaluation Campaign - machine translation track) 
    - http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html
- Several reports have indicated that the performance of punctuation prediction is largely influenced by the average number of punctuation marks per utterance in the dataset (Wang et al., 2012).
    - It is quite understandable that there
      will always be a period or question mark at the end of the
      utterance. In their evaluation, they remove all the segmentation from both the reference and ASR transcripts of IWSLT data files and there is only one single utterance existing in each dataset.
- GloVe embeddings (50 dim)
- Uses CNNs
