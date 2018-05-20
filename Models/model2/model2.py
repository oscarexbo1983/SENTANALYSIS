from helpers import *
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report


def scikit_TFIDF(m, n, Total_clean_train, Total_clean_test):
    print('Vectorizing with TFIDF')
    vectorizer = CountVectorizer(min_df=1, max_features=m, ngram_range=(1, n))
    vectorizer.build_analyzer
    X_train = vectorizer.fit_transform(Total_clean_train).toarray()
    X_test = vectorizer.transform(Total_clean_test).toarray()
    transformer = TfidfTransformer()
    tfidf_train = transformer.fit_transform(X_train).toarray()
    tfidf_test = transformer.transform(X_test).toarray()
    return tfidf_train, tfidf_test


def tfidf_vectorizer(tweets_filename, features):
    if isinstance(tweets_filename, str):
        tweets = helpers.read_text(tweets_filename)
    elif isinstance(tweets_filename, list):
        tweets = tweets_filename

    tokenize = lambda doc: doc.lower().split(" ")

    tfidf = TfidfVectorizer(norm='l1', max_features=features, min_df=0, use_idf=True, smooth_idf=False,
                            sublinear_tf=True, tokenizer=tokenize)
    tweets_rep = tfidf.fit_transform(tweets)

    return tweets_rep


print("Loading Positive Tweets..")
new_positive = open("preprocessed/pre_positive_v2.txt", 'rb')
complete_tweets = []
i = 0
pos = 0
for line in new_positive:
    if pos < 100000:
        i = i + 1
        pos = pos + 1
        line = line.decode('utf8')
        line = re.sub("\s\s+", " ", line)
        complete_tweets.append(line)
print('DONE..')

print("Loading Negative Tweets..")
new_negative = open("preprocessed/pre_negative_v2.txt", 'rb')
j = 0
neg = 0
for line in new_negative:
    if neg < 100000:
        j += 1
        line = line.decode('utf8')
        line = re.sub("\s\s+", " ", line)
        neg = neg + 1
        complete_tweets.append(line)
print('DONE..')

print("Putting together Positive and Negative Tweets")
indexes = np.arange(0, len(complete_tweets))
np.random.shuffle(indexes)
y_pos = [1] * pos
y_neg = [0] * neg

train_labels = y_pos + y_neg
train_labels = np.array(train_labels)
train_labels = train_labels[indexes]

train_dataset = []
for i in indexes:
    train_dataset.append(complete_tweets[i])

print("Loading Test Tweets..")
test_tweets = open("preprocessed/pre_test_v2.txt", 'rb')
test_dataset = []
j = 0
for line in test_tweets:
    j += 1
    line = line.decode('utf8')
    line = re.sub("\s\s+", " ", line)
    test_dataset.append(line)
print('DONE..')
labels = train_labels

##TFDIF Vectorization
X_train, X_test = scikit_TFIDF(1200, 4, train_dataset, test_dataset)
# X_train=tfidf_vectorizer()
print("Done..")

##Split training to see accuracy
train_x = X_train[0:180000]
test_x = X_train[180001:]
labels_train = labels[0:180000]
labels_test = labels[180001:]

print("Training with SVM")
lin_clf = svm.LinearSVC()
lin_clf.fit(train_x, labels_train)
Y_SVC_linear_predic = lin_clf.predict(test_x)
print("Done")

print(classification_report(Y_SVC_linear_predic, labels_test))

print("Training with Bayes")
gnb = GaussianNB()
Y_pred_Bayes = gnb.fit(train_x, labels_train).predict(test_x)
print("Done")

print(classification_report(Y_pred_Bayes, labels_test))

print("Preparing data for submission..")
print("Training with SVM")
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, labels)
predicted_SVM = lin_clf.predict(X_test)
print("Done")
predicted_SVM[predicted_SVM == 0] = -1

print('Creating final csv submission file SVM')
submission_csv(predicted_SVM, 'submission_TFIDF_SVM.csv')
print("File created..")

print("Training with Bayes")
gnb = GaussianNB()
predicted_Bayes = gnb.fit(X_train, labels).predict(X_test)
print("Done")
predicted_Bayes[predicted_Bayes == 0] = -1

print('Creating final csv submission file Bayes')
submission_csv(predicted_Bayes, 'submission_TFIDF_Bayes.csv')
print("File created..")
