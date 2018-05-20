# Model 1 (Word vectors - SVM)

## Getting Started

Before continue with these steps, all steps from main README have to be accomplish
        
## Download Datasets and Pretrained Data

1. Open a browser in https://drive.google.com/open?id=1ldH8u0YVvhIpBpqgjIyD1fT_jxivHr3P
2. Right click on model1 folder and click on Download
3. Go to your downloads directory and unzip the file
4. Copy the contents of the unzipped folder (More specifically 'wordvectors' folder) to model1's root path
5. If it is necessary, replace the old folders with the new downloaded folders.

NOTE: This models uses preprocessed datasets, for more details please refer to preprocessing folder in root path.

## Running model 2

1. Change directory to model2

```
cd Models/model1
```

2. Run Model

```
python3 model1.py
```

It will create a file in model's root path called "submission_model2.csv"

## Generate pretrained data

1. Open model1.py and update line 8, change the flag to False

```
USE_PRETRAINED_MODEL = False (Previously in True)
```

## Running Times

* ~10 minutes to run the classifier with a pretrained model.
* ~30 minutes to run the classifier without a pretrained model.