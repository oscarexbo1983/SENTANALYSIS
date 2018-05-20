import helpers
import data
import xgboost as xgb
import numpy as np
import _pickle as cPickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Embedding

USE_PRETRAINED_MODEL = True

def print_shapes():
    print("Y_TRAIN: ", y_train.shape)
    print("X_TRAIN: ", x_train.shape)
    print("X_TEST: ", x_test.shape)

# Load Data
size = 1200000

if USE_PRETRAINED_MODEL:
    print("Loading Pretrained Data")

    x_train = cPickle.load(open("datasaved/xtrainCNNpretrained.pkl", "rb"))
    x_test = cPickle.load(open("datasaved/xtestCNNpretrained.pkl", "rb"))
else:
    print("Loading Data")

    tweets_dict = helpers.join_files({ "test": ["data/test_data.txt"],
                                       "train": ["data/train_pos_full.txt", "data/train_neg_full.txt"] }, size)

    # Get Data Ready to be trained by model
    print("Loading Numeric Representation")

    x_train, y_train, x_test, max_features, tokenizer, ngram_rep = data.get_featured_data(tweets_dict["train"], tweets_dict["test"], True)

    print("Keras Model")

    np.random.seed(7)

    model = Sequential()
    model.add(Embedding(max_features + 1, 50, input_length=x_train.shape[1]))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Model Summary")

    print(model.summary())

    print("Training with Keras Model")

    model.fit(x_train, y_train, validation_split=0.1, epochs=1, batch_size=128, verbose=1, shuffle=True)

    x_train = model.predict_proba(x_train, batch_size=128)
    x_test = model.predict_proba(x_test)

    cPickle.dump(x_train, open('datasaved/xtrainmodel5.pkl', 'wb'))
    cPickle.dump(x_test, open('datasaved/xtestmodel5.pkl', 'wb'))

print("Training with XBG Classifier")

y = np.array(size * [0] + size * [1])
np.random.seed(0)
np.random.shuffle(y)

model = xgb.XGBClassifier().fit(x_train, y)
y_pred = model.predict(x_test)

y_pred = 1 - 2 * y_pred

helpers.submission_csv(y_pred, "submit")

print("Done")
