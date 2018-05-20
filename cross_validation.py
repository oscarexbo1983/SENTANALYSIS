import helpers
import data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Embedding
from sklearn.model_selection import StratifiedKFold


size = 600000
tweets_dict = helpers.join_files({ "test": ["data/test_data.txt"],
                                   "train": ["data/train_pos_full.txt",
                                             "data/train_neg_full.txt"] },
                                 size)

x_train, y_train, x_test, max_features, tokenizer, ngram_rep = data.get_featured_data(tweets_dict["train"], tweets_dict["test"], True)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []
iteration = 0

np.random.seed(7)

for train, test in kfold.split(x_train, y_train):
    model = Sequential()
    model.add(Embedding(max_features + 1, 50, input_length=x_train[train].shape[1]))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train[train], y_train[train], validation_split=0.1, epochs=1, batch_size=8096, verbose=0, shuffle=True)

    # evaluate the model
    scores = model.evaluate(x_train[test], y_train[test], verbose=0)
    print("Validation Accuracy %s: %.2f%%" % (iteration, scores[1] * 100))

    cvscores.append(scores[1] * 100)

    iteration += 1
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
