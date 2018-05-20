import pickle
import numpy as np
import xgboost as xgb
import helpers


print("Loading probability vectors of models 4 and 5...")
train1 = pickle.load(open("train_features/xtrain1epNN.pkl", "rb"))
test1 = pickle.load(open("test_features/xtest1epNN.pkl", "rb"))
train2 = pickle.load(open("train_features/xtrain1epCNN.pkl", "rb"))
test2 = pickle.load(open("test_features/xtest1epCNN.pkl", "rb"))

size = int(len(train1) / 2)
print("Building the complete probability train and test matrices..")
train = np.hstack((train1, train2))
test = np.hstack((test1, test2))

print("Training with XGBoost Classifier..")
y = np.array(size * [0] + size * [1])
np.random.seed(0)
np.random.shuffle(y)

model = xgb.XGBClassifier().fit(train, y)
y_pred = model.predict(test)
print("Done")

y_pred = 1 - 2 * y_pred

print("Creating submission file..")
helpers.submission_csv(y_pred, "submission_model6")
print("File created..")
