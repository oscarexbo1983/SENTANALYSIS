# Model 4 (Bigram - NN with pretrained data)

## Getting Started

Before continue with these steps, all steps from main README have to be accomplish
        
## Download Datasets and Pretrained Data

1. Open a browser in https://drive.google.com/open?id=1ldH8u0YVvhIpBpqgjIyD1fT_jxivHr3P
2. Right click on model4 folder and click on Download
3. Go to your downloads directory and unzip the file
4. Copy the contents of the downloaded folder (More specifically 'data' and 'embeddings'folder) to model4's root path
5. If it is necessary, replace the old folders with the new downloaded folders.

## Running model 4

1. Change directory to model4

```
cd Models/model4
```

2. Run Model

```
python3 model4.py
```

It will create a file in model's root path called "submission_model4.csv"

## Generate pretrained data

1. Open model4.py and update line 10, change the flag to False

```
USE_PRETRAINED_MODEL = False (Previously in True)
```
