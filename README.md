# neural-punctuator
Complimentary code for our paper **_Automatic punctuation restoration with BERT models_** submitted to the XVII. Conference on Hungarian Computational Linguistics.

## Abstract
We present an approach for automatic punctuation restora-tion with BERT models for English and Hungarian. For English, weconduct our experiments on Ted Talks, a commonly used benchmark forpunctuation restoration, while for Hungarian we evaluate our models onthe Szeged Treebank dataset. Our best models achieve a macro-averagedF1-score of 79.8 in English and 82.2 in Hungarian

## Repository Structure

```
.
|-- docs
|   └── paper       # The submitted paper
|-- notebooks       # Notebooks for data preparation/preprocessing
|-- src
    └── neural_punctuator 
        ├── base            # Base classes for training Torch models
        ├── configs         # YAMl files defining the parameters of each model
        ├── models          # Torch model definitions
        ├── preprocessors   # Preprocessor class
        ├── trainers        # Train logic
        ├── utils           # Utility scripts (logging, metrics, tensorboard etc.)
        └── wrappers        # Wrapper classes for the models containing all the components needed for training/prediction
```
## Dataset
Ted Talk dataset (English) - http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html

Szeged Treebank (Hungarian) - https://rgai.inf.u-szeged.hu/node/113


## Authors
Attila Nagy, Bence Bial, Judit Ács

Budapest University of Technology and Economics - Department of Automation and Applied Informatics
