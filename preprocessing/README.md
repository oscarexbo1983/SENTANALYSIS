# Preprocessing

All preprocessing methods are at this folder. Datasets are taken from 'data' folder in root's path. Main files are 
'preprocessing.py' and 'prepare_data.py'. Preprocessing.py has all the preprocessing methods that we use in our tests.
Prepare_data.py is a python file where you can apply these preprocess methods to raw datasets.

## Prerequisites

1. Install NLTK data
```
python -m nltk.downloader all
```

## Getting Started

Before continue with this steps all steps from main README have to be accomplish
        
## Generating a preprocess dataset

1. Change directory to preprocessing

```
cd preprocessing
```

2. Run prepare_data.py

```
python3 prepare_data.py
```

This program will take raw datasets inside 'data' folder in root path, preprocess it and save it into 'preprocessed' folder

## Add any preprocess method from preprocessing file

1. Open prepare_data.py and add methods of preprocessing file on line 74 

