import pickle
import numpy as np
from helpers import *
from sklearn import svm
from sklearn.metrics import classification_report
from generate_embeddings import *

USE_PRETRAINED_MODEL = True

if USE_PRETRAINED_MODEL==False:
    generate_embeddings()

elif USE_PRETRAINED_MODEL==True:
    ##Load vocab and embeddings
    print("Loading vocabulary and Word Embeddings...")
    with open('wordvectors/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    embeddings = np.load('wordvectors/embeddings.npy')

    print("Loading Positive Tweets vector..")
    with open('positivevector.pkl', 'rb') as f:
        positive = pickle.load(f)
    print("Loading Negative Tweets vector..")
    with open('negativevector.pkl', 'rb') as f:
        negative = pickle.load(f)
    print("Loading Test Tweets vector..")
    with open('testvector.pkl', 'rb') as f:
        test_set = pickle.load(f)

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
    submission_csv(predicted_SVM, 'submission_model1.csv')
    print("File created..")
