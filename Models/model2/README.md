# Model 2 (TFIDF - SVM/Naive Bayes)

## Getting Started

Before continue with these steps, all steps from main README have to be accomplish
        
## Download Datasets and Pretrained Data

1. Open a browser in https://drive.google.com/open?id=1ldH8u0YVvhIpBpqgjIyD1fT_jxivHr3P
2. Right click on model2 folder and click on Download
3. Go to your downloads directory and unzip the file
4. Copy the contents of the unzipped folder (More specifically 'preprocessed' folder) to model2's root path
5. If it is necessary, replace the old folders with the new downloaded folders.

NOTE: This models uses preprocessed datasets, for more details please refer to preprocessing folder in root path.

## Running model 2

1. Change directory to model2

```
cd Models/model2
```

2. Run Model

```
python3 model2.py
```

It will create two files in model's root path named "submission_TFIDF_Bayes.csv" and "submission_TFIDF_SVM.csv"

## Running Times

* ~5 minutes to run the classifier.

