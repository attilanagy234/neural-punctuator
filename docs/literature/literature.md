# Notes on literature
---

### Punctuation Prediction Model for Conversational Speech
[Paper]

https://arxiv.org/pdf/1807.00543.pdf
 - **Dataset:** Fischer corpus (dialogues)
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
[Paper]

http://www.lrec-conf.org/proceedings/lrec2016/pdf/103_Paper.pdf
- Treats the classification problem as whether a word in the sequence is followed by a punctuation mark.
- Classify whether a punctuation mark should be inserted after the third word of a 5-words sequence and which kind of punctuation mark the inserted one should be.
- Gives a nice literature overview --> we could refer to these papers in our own literature overview.
- 4 classes in total: O (means no punctuation mark followed), COMMA, PERIOD and QUESTION.
- Exclamation marks or semicolons are classified as PERIOD, while colons or dashes are classified as COMMA. There are no brackets in our data and all the other punctuation marks, such as quotation marks, are just ignored.
- **Dataset:** TED Talks (IWSLT 2012 Evaluation Campaign - machine translation track) 
    - http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html
- Several reports have indicated that the performance of punctuation prediction is largely influenced by the average number of punctuation marks per utterance in the dataset (Wang et al., 2012).
    - It is quite understandable that there
      will always be a period or question mark at the end of the
      utterance. In their evaluation, they remove all the segmentation from both the reference and ASR transcripts of IWSLT data files and there is only one single utterance existing in each dataset.
- GloVe embeddings (50 dim)
- Uses CNNs

### BertPunc
[Code]

https://github.com/nkrnrnk/BertPunc

### Using bidirectional LSTM with BERT for Chinese punctuation prediction
[Paper]

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9172986&tag=1

### Adversarial Transfer Learning for Punctuation Restoration
[Paper]

https://arxiv.org/pdf/2004.00248.pdf

### Efficient Automatic Punctuation Restoration Using Bidirectional Transformers with Robust Inference
[Paper] [SOTA]

https://www.aclweb.org/anthology/2020.iwslt-1.33.pdf

### Deep Recurrent Neural Networks with Layer-wise Multi-head Attentions for Punctuation Restoration
[Paper]

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8682418

### Overview of the IWSLT 2012 Evaluation Campaign
[Paper]

http://hltc.cs.ust.hk/iwslt/proceedings/overview12.pdf
 - Summary on TED talk dataset

--
## Papers by Hungarian authors that we should check (maybe they used hungarian datasets that we could compare to):
 - [x] http://acta.uni-obuda.hu/Szaszak_89.pdf
 - [x] https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2132.pdf
   - Mentions https://www.researchgate.net/publication/279186001_Automatic_Close_Captioning_for_Live_Hungarian_Television_Broadcast_Speech_A_Fast_and_Resource-Efficient_Approach in data chapter, but could not find data (yet)
     
 - [ ] https://www.researchgate.net/publication/327977369_Joint_Word-and_Character-level_Embedding_CNN-RNN_Models_for_Punctuation_Restoration
 - [ ] https://ieeexplore.ieee.org/document/8268227
 
## Potential dataset to use:
- https://rgai.inf.u-szeged.hu/node/128
- Babel: https://www.researchgate.net/publication/221481223_BABEL_An_Eastern_European_multi-language_database
- Broadcast news: https://www.researchgate.net/publication/221487490_The_COST278_broadcast_news_segmentation_and_speaker_clustering_evaluation_-_overview_methodology_systems_results

