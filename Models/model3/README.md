# Model 3 (Pretrained word embeddings - CNN, Tensorflow)

## Getting Started

Before continue with these steps, all steps from main README have to be accomplish
        
## Download Datasets and Pretrained Data

1. Open a browser in https://drive.google.com/open?id=1ldH8u0YVvhIpBpqgjIyD1fT_jxivHr3P
2. Right click on model3 folder and click on Download, it should download a zip file
3. Right click on 'glove.twitter.27B.200d.txt' file and click on Download
4. Go to your downloads directory and unzip the first downloaded zip file, it should create a folder named model3
5. Copy file 'glove.twitter.27B.200d.txt' ,downloaded last, to the next path
```
{Download folder}/model3/data/
```
6. Copy the contents of unzipped model3 folder (More specifically 'data','preprocessed' and 'runs' folders) to our project model3's root path
7. If it is necessary, replace the old folders with the new downloaded folders.

## Running model 3

1. Change directory to model3

```
cd Models/model3
```

2. Run Model

```
python3 model3.py
```

It will create a file in model's root path called "submission_model3.csv"

## Generate pretrained data

1. Open model3.py and update line 14, change the flag to False

```
USE_PRETRAINED_MODEL = False (Previously in True)
```

## Running Times

* ~10 minutes to run the classifier with a pretrained model.
* ~2 hours to run the classifier without a pretrained model.
