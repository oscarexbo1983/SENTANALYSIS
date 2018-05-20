# Data Mining II

 The task of this work is to predict if a tweet message used to contain a positive  :) or negative :( smiley,
 by considering only the remaining text.
Best Model (Model5) has its files in the root path path and another models used for research are in Models folder.
Project Structure is completely explained at the end of this README.
  
## Getting Started

1. Verify that you have project2.zip file on MacOS, Windows or Linux System.

### Prerequisites

1. These steps were tested using Macbook Pro with High Sierra Operative Systems, therefore if you test with that system these steps must work.
   
   NOTE: XGBoost installation was achieved using another guide that is described in next steps.
         
   NOTE: Windows environments do not work well with NN and CNN frameworks, we tried to install on these systems,
    but it takes a lot of time and in some frameworks the installation is not complete.
   
   NOTE: These steps should work for any UNIX machine, maybe it may have some different steps to accomplish depending the OS used (e.g. Another MacOS version or Linux),
    principally the installation of XGBoost.
        
2. Anaconda.
    1.  For MacOs you can install using GUI or just using Terminal commands

        ```
        Use instructions from: https://docs.anaconda.com/anaconda/install/mac-os#macos-graphical-install
        ```
    
    2. For Windows
          
        ```
        Use instructions from: https://docs.anaconda.com/anaconda/install/windows
        ```
    3. Verify Anaconda installation, Run from Terminal
    
        ```
        python
        ```
       
       This should appear:
        
        ```
        Python 3.6.2 |Anaconda, Inc.| (default, Sep 21 2017, 18:29:43)
        [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
        Type "help", "copyright", "credits" or "license" for more information.
        ```
        
        Or at least the "Anaconda" and "3.6" keywords should be present in interpreters definition
        
3. TensorFlow, XGBoost, Keras, Scikit and other prerequisistes:
    1.  Run:
    
        ```
        pip install -r prerequisites.txt
        ```
        
    2. (Optional) If you experience some problems installing XGBoost, principally in MacOS High Sierra
     you can use the tutorial from the web page [Install XGBoost](https://ampersandacademy.com/tutorials/python-data-science/install-xgboost-on-mac-os-sierra-for-python-programming)
        
        These are some steps to follow (Taken from the resource above)
        You need to have git and homebrew installed (https://brew.sh; https://git-scm.com/book/en/v1/Getting-Started-Installing-Git)
        1. brew install gcc --without-multilib (~ 30 minutes)
        
        2. git clone --recursive https://github.com/dmlc/xgboost

        3. cd xgboost; cp make/config.mk ./config.mk

        4. vi config.mk
        5. Uncomment the lines near the top of the file:

        ```
        export CC = gcc
        export CXX = g++
       
        Change them to the following:

        export CC = gcc-7
        export CXX = g++-7
         ```
        
        6. Save the file. Also, make changes to the file xgboost/Makefile. Open the file using vi or text editor.
        
        ```
        vi Makefile
        ```
        
        7. And Change them to the following in the Makefile.
        
        ```
        export CC = gcc-7
        
        export CXX = g++-7
        ```
        
        8. Save the file. Now you need to run a cleaning step since you changed the Makefile.
        
        ```
        make clean_all && make -j4
        ```
    
        9. Python Package Installation

        ```
        cd python-package; sudo python setup.py install
        ```
        
### Installing

1. Open Terminal and go to project file zip file

```
cd { $project_file_path }
```

2. Unzip project2.zip file

```
unzip -a project2.zip
```

## Download Datasets and Pretrained Data

1. Open a browser in https://drive.google.com/open?id=1ldH8u0YVvhIpBpqgjIyD1fT_jxivHr3P
2. Right click on model5 folder and click on Download
3. Go to your downloads directory and unzip the file
4. Copy the contents of the downloaded folder (More specifically 'data' folder) to project's root path
5. If it is necessary, replace the old folders with the new downloaded folders.

## Running our best prediction model

1. Go inside project's directory

```
cd SAD
```

2. Run Model

```
python3 run.py
```

It will create a file in project path called "submit.csv", that file can be uploaded to [Kaggle](https://www.kaggle.com/c/epfml17-text/submit).

## Generate pretrained data

1. Open run.py and update line 9, change the flag to False

```
USE_PRETRAINED_MODEL = False (Previously in True)
```

## Running Times

* ~10 minutes to run the classifier with a pretrained model.
* ~12 hours to run the classifier without a pretrained model.

## Cross Validation for Best Model

1. Run Cross Validation file in the root path

```
python3 cross_validation.py
```

## Run other model

1. Change directory to each model you want to test

```
cd Models/{model1, model2, model3...modeln}
```

2. Read its README file and continue from there

## Problems
If you experience any problems downloading datasets or any confusion running models, you can download the full project's folder on:

```
https://drive.switch.ch/index.php/s/FN5dmSE30cd0tc1
```

## Project Structure

The project core has the next structure:
 * Models
     * model1 (Word Vectors + SVM)
        * data (Small Dataset, train positive, train negative and test)
            * neg_train_txt
            * pos_train.txt
            * test_data.txt
        * wordvectors 
            * Initial files given to generate python files for model1
        * helpers.py 
            * Helper methods: load txt methods, "numerical representation" methods and csv methods.
        * model1.py
            * Main Python Script for model1, contains all the process to output the model1 prediction
        * negativevector.pkl
            * Pickle file for negative tweets
        * positivevector.pkl
            * Pickle file for positive tweets
        * testvector.pkl
            * Pickle file for test tweets
            
     * model2 (TFIDF - SVM/Naive Bayes)
        * preprocessed (Dataset with preprocessing, train positive, train negative and test)
        * helpers.py
            * Helper methods: load txt methods, tweet's helper method and csv methods.
        * model2.py
            * Main Python Script for model2, contains all the process to output the model2 prediction
        * CSV files
            * CSV submission files
            
     * model3 (Pretrained word embeddings - CNN, Tensorflow)
        * data (Pretrained datasets: Word2vec and GloVe)
        * preprocessed (Dataset with preprocessing, train positive, train negative and test)
        * runs (Folders with tensorflow's checkpoints )
        * config.yml
            * Configuration File with some file path configurations
        * data_helpers.py
            * Helper methods: load pretrained methods, cleaning tweets methods and loading data methods.
        * evaluation.py
            * Load trained model from run folder and output predict data.
        * helpers.py
            * Helper methods: load txt methods, tweet's helper method and csv methods.
        * model3.py
            * Main Python Script for model3, contains all the process to output the model3 prediction
        * text_cnn.py
            * All layers definitions
        * CSV files
            * CSV submission files
        * PNG files 
            * TensorBoard CNN images
            
     * model4 (Bigram - NN with pretrained data)
        * data
        * datasaved
        * data.py
        * helpers.py
        * model4.py
            * Main Python Script for model4, contains all the process to output the model4 prediction
        
     * model6 (Combination of CNN and NN models)
        * test_features (Saved probabilistic models for test datasets)
        * train_features (Saved probabilistic models for train datasets)
        * helpers.py
            * Helper methods: load txt methods, tweet's helper method and csv methods.
        * model6.py
            * Main Python Script for model6, contains all the process to output the model6 prediction
        * CSV file
             * CSV submission file
 * preprocessing
    * preprocessed (Folder where files preprocessed will be generated)
    * big.txt
        Text properly written to check spelling
    * prepare_data.py
        * Python file that allows to preprocess datasets
    * preprocessing.py
        * Preprocessing python file, it contains all the preprocessing methods
 
 * data (Full Dataset without any preprocessing, train_neg_full, train_pos_full and test_data)
 * datasaved (Folder to save probabilistic models)
 * cross_validation.py 
    * Cross Validation code for model 5
 * data.py
    * Helper methods to get data ready for model
 * helpers.py
    * General helper methods to read files get bigrams and more
 * run.py
    * Main Python Script for Project 2, contains all the process to output best prediction for Kaggle competition
 * prerequisites.txt
    * Prerequisites file, it contains all packages to be installed using pip
 * README.md (Project README File)

## Contributing

To contribute, look at the project structures, it has files and roles of files well defined. In order to reuse the code
those files can be updated with new methods that correspond to the same class as the file.

## 