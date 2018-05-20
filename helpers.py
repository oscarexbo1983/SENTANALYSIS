import numpy as np
import csv
import pickle
from scipy.sparse import *
import nltk


def read_text(filename, lineslimit=False, preprocess=False):
    """
    DESCRIPTION: Returns list data with all tweets from file
    INPUT:
            filename: Name of the file to extract tweets
            lineslimit: Number of lines to limit the read of the file
            preprocess: Boolean to use preprocess
    OUTPUT:
            data: List of tweets from file
    """

    data = []

    with open(filename, "r", encoding="utf-8") as f:
        pos = 0

        for line in f:
            if lineslimit and pos < lineslimit:
                # if preprocess:
                # line = clean(line)
                data.append(line)
            else:
                break
            pos = pos + 1

    return data


def read_list_text(list_files, lineslimit=False, preprocess=False):
    """
    DESCRIPTION: Returns a list of lists with all tweets from a list of files
    INPUT:
            list_files: List of filenames to get their tweets
            lineslimit: Number of lines to limit on the read of each file
            preprocess: Boolean to use preprocess
    OUTPUT:
            list_doc: List of Lists of tweets from the lists of files
    """

    list_doc = []

    for files in list_files:
        list_doc.append(read_text(files, lineslimit, preprocess))

    return list_doc


def join_files(list_of_files, lineslimit=False, preprocess=False):
    """
    DESCRIPTION: Returns a dictionary with a list of tweets from files
    INPUT:
            list_of_files: Name of the file to extract tweets
            lineslimit: Number of lines to limit the read of the file
            preprocess: Boolean to use preprocess
    OUTPUT:
            Python Dictionary with two keys "Test" and "Train" and a list of tweets referencing to each dataset as values.
    """

    test_files = []
    train_files = []
    test_tweets = []

    for file in list_of_files["test"]:
        for tweet in read_text(file, lineslimit, preprocess):
            tweet = tweet[tweet.find(',') + 1:]
            test_tweets.append(tweet)
        test_files = sum([test_files, test_tweets], [])

    train_tweets = read_list_text(list_of_files["train"], lineslimit, preprocess)

    for train_tweet in train_tweets:
        train_files = sum([train_files, train_tweet], [])

    return {"test": test_files, "train": train_files}


def submission_csv(y_pred, filename):
    """
    DESCRIPTION:
            Creates the final submission file to be uploaded on Kaggle platform
    INPUT:
            y_pred: List of sentiment predictions. Contains 1 and -1 values
    """

    with open(filename + ".csv", "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        r1 = 1

        for r2 in y_pred:
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
            r1 += 1


def get_ngram(seq, n=2):
    """
    DESCRIPTION: Returns ngrams from a sequence of objects
    INPUT:
            seq: Sequence of Numbers, in this case indexes that references to vocabulary
            n: Number to use to take contiguous sequences of words
    OUTPUT:
            Python Set with contiguous sequences of words
    """

    return set(zip(*[seq[i:] for i in range(n)]))


def get_bigram(seq):
    """
    DESCRIPTION: Returns ngrams from a sequence of objects
    INPUT:
            seq: Sequence of Numbers, in this case indexes that references to vocabulary
    OUTPUT:
            Python Set with contiguous sequences of 2 words, e.g. get_bigram([1,2,3,4])) => {(1, 2), (3, 4), (2, 3)}
    """

    return set(nltk.bigrams(seq))


def get_ngram_seq(sequences, ngram_rep):
    """
    DESCRIPTION: Returns sequences of tweets with its ngram extension
    INPUT:
            sequence: Sequences of tweets with numeric representation without ngram extension
            ngram_rep: Ngram Vocabulary
    OUTPUT:
            ngram_sequences: Sequences of tweets with its ngram extension
    """

    print("Adding ngram extension to sequences")

    ngram_sequences = []

    for seq in sequences:
        ngram_incl = seq[:]

        for index, value in enumerate(seq):
            ngram_key = tuple(seq[index: index + 2])

            if ngram_key in ngram_rep:
                ngram_incl.append(ngram_rep[ngram_key])

        ngram_sequences.append(ngram_incl)

    return ngram_sequences
