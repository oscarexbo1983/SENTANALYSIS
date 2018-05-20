import os
import pickle
import numpy as np
from helpers import *
from sklearn import svm
from sklearn.metrics import classification_report
from generate_embeddings import *

def load_word_matrix(vocab, train_filename, embeddings):
    tweets_array = np.zeros(shape=(1, 20))
    with open(train_filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            #print(idx)
            words = line.strip().split(" ")
            
            tweet_array = []
            
            for word in words:
                if word in vocab:
                    index = vocab[word]
                    tweet_array.append(embeddings[index])
        
            if len(tweet_array) != 0:
                mean_vector = np.mean(np.array(tweet_array), axis=0)
            else:
                mean_vector = np.zeros(shape=(1, 20))
            
            tweets_array = np.concatenate((tweets_array, mean_vector.reshape(1, 20)))
            
            if idx % 5000 == 0:
                print("Tweet number {i}".format(i=idx))
    return tweets_array[1:]


def generate_embeddings():
    owd=os.getcwd()
    wordvectors=owd+'/wordvectors'
    print(wordvectors)
    #os.getcwd()
    os.chdir(wordvectors)
    print("Creating Vocabulary..")
    os.system('sh build_vocab.sh')
    #os.getcwd()
    print("Vocabulary Created..")
    print("Creating Cut Vocabulary..")
    os.system('sh cut_vocab.sh')
    print("Vocabulary Cut..")
    os.system('python3 pickle_vocab.py')
    print("Vocab pkl generated..")
    os.system('python3 cooc.py')
    print("Coocurrence matrix created..")
    os.chdir(owd)
    os.chdir(wordvectors)
    os.system('python3 glove_sgd.py')
    print("Embeddings created..")
    os.chdir(owd)
    data=np.load('wordvectors/embeddings.npy')
    with open('wordvectors/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    embeddings = np.load('wordvectors/embeddings.npy')

    ##Word Vectors for Positive Tweets
    path_tweets="wordvectors/pos_train.txt"

    print("Building Vector for Positive Tweets...")
    positive = load_word_matrix(vocab, path_tweets, embeddings)

    ##Word Vectors for Negative Tweets
    path_tweets="wordvectors/neg_train.txt"

    print("Building Vector for Negative Tweets...")
    negative = load_word_matrix(vocab, path_tweets, embeddings)
    print("DONE")

    ##Word Vectors for Test Set
    path_tweets="wordvectors/test_data.txt"

    print("Building Vector for Test Set...")
    test_set = load_word_matrix(vocab, path_tweets, embeddings)
    print("DONE")

    # Putting both together and adding labels
    ones_p = np.ones(len(positive))
    zeros_n = np.zeros(len(negative))
    
    complete_pos = np.column_stack((positive, ones_p))
    complete_neg = np.column_stack((negative, zeros_n))
    train_matrix = np.row_stack((complete_pos, complete_neg))
    X = train_matrix
    
    new_train_matrix = np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)
    train_dataset = new_train_matrix[0:train_matrix.shape[0], 0:-1]
    train_labels = new_train_matrix[0:train_matrix.shape[0], -1]
    
    print("Train shape:", train_dataset.shape)
    print("Train labels:", train_labels.shape)
    print("Test set:", test_set.shape)
    
    
    ##Split training to see accuracy 10% for validation data
    a = int(len(train_dataset) * 0.9)
    b = a + 1
    X_train = train_dataset
    labels = train_labels
    train_x = X_train[0:a]
    test_x = X_train[b:]
    labels_train = labels[0:a]
    labels_test = labels[b:]
    
    print("Training with SVM Classifier to get model accuracy..")
    lin_clf = svm.LinearSVC()
    lin_clf.fit(train_x, labels_train)
    Y_SVC_linear_predic = lin_clf.predict(test_x)
    print("Done")
    print(classification_report(Y_SVC_linear_predic, labels_test))
    
    print("Training with SVM with complete Train Set")
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, labels)
    predicted_SVM = lin_clf.predict(test_set)
    print("Done")
    
    predicted_SVM[predicted_SVM == 0] = -1
    print("Creating final csv submission file")
    submission_csv(predicted_SVM, 'submission.csv')
    print("File created..")







