import helpers
import xgboost as xgb
import numpy as np
import data
import _pickle as cPickle
from keras.models import Sequential
from keras.layers import GlobalAveragePooling1D, Dense, Embedding


USE_PRETRAINED_MODEL = True

def print_shapes():
    print("Y_TRAIN: ", y_train.shape)
    print("X_TRAIN: ", x_train.shape)
    print("X_TEST: ", x_test.shape)

# Load Data
size = 300000

if USE_PRETRAINED_MODEL:
    print("Loading Pretrained Data")

    x_train = cPickle.load(open("datasaved/xtrainNNpretrained.pkl", "rb"))
    x_test = cPickle.load(open("datasaved/xtestNNpretrained.pkl", "rb"))

else:

    print("Loading Data")

    tweets_dict = helpers.join_files({ "test": ["data/test_data.txt"],
                                       "train": ["data/train_pos_full.txt",
                                                 "data/train_neg_full.txt"] }, size)

    # Get Data Ready to be trained by model
    print("Loading Numeric Representation")

    x_train, y_train, x_test, max_features, tokenizer, ngram_rep = data.get_featured_data(tweets_dict["train"], tweets_dict["test"], True)

    # ngram_rep.update(tokenizer.word_index)

    max_features_pretrained, W = data.get_pretrained_data_without_ngram(ngram_rep, max_features)

    print("Keras Model")
    model = Sequential()
    model.add(Embedding(np.max((max_features, max_features_pretrained)) + 1, 200, input_length=x_train.shape[1], weights=[W]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #calculate predictions
    model.fit(x_train, y_train, validation_split=0.1, epochs=1, batch_size=128, verbose=1)
    x_train = model.predict_proba(x_train, batch_size=64)
    x_test = model.predict_proba(x_test)

    cPickle.dump(x_train, open('datasaved/xtrainmodel4.pkl', 'wb'))
    cPickle.dump(x_test, open('datasaved/xtestmodel4.pkl', 'wb'))

print("Training with XBG Classifier")
y = np.array(size * [0] + size * [1])
np.random.seed(0)
np.random.shuffle(y)

model = xgb.XGBClassifier().fit(x_train, y)
y_pred = model.predict(x_test)

y_pred = 1 - 2 * y_pred

helpers.submission_csv(y_pred, "submission_model4")

print("Done")