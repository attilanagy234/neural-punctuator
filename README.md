# neural-punctuator
Predicting punctuations with neural networks solely based on their textual representation.

## Structure

```
.
|-- docs
|   └── literature  # Short summaries on the papers/sources that we have read
|   └── paper       # The submitted paper for the homework
|-- notebooks       # Notebooks for EDA, preprocessing, model explorations and experiments
|-- src
    └── neural_punctuator 
        ├── base            # Base classes for training Torch models
        ├── configs         # YAMl files defining the parameters of each model
        ├── models          # Contains definitions of different Torch models
        ├── preprocessors   # Contains model-specific preprocessor classes
        ├── trainers        # Contains the train logic for every model
        ├── tuners          # Contains hyperparameter optimizers
        ├── utils           # Utility scripts
        ├── wrappers        # Wrapper classes for the models containing all the components needed for training/prediction
        └── main.py         # Entry point
```
## Dataset
Ted talk dataset - Acquired from here: https://zenodo.org/record/4061423?fbclid=IwAR3IkZedUbnMtPrPdmkZJXUITd_AOzluiz53dhkR_Yc01TzbOpGo9TBjDBE#.X4sNB9AzaUk

## How to run
TODO

## Authors
Bence Bial, Attila Nagy

Budapest University of Technology and Economics
