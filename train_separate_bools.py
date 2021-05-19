import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pickle
import time
import seaborn as sns

start_time = time.time()
#Load data
print("Reading data...")
data = pd.read_csv("feature_engineering.csv")
print(data.head(5))

data = data.drop(columns=["prob_booked","prob_clicked","date_time","position"])
data.replace([np.inf, -np.inf], int(0), inplace=True)

#normalize data
min_max_scaler = MinMaxScaler()
X,y = data.drop(["click_bool","booking_bool"],axis=1), data.loc[:,["click_bool","booking_bool"]]
X[['price_usd',"price_difference_user"]] = min_max_scaler.fit_transform(X[['price_usd',"price_difference_user"]])
# print(X.columns)
print(len(X.columns))

X_train, X_test, Y_train, Y_test = train_test_split(X, y.loc[:,"click_bool"], test_size=0.1, random_state=0)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, y.loc[:,"booking_bool"], test_size=0.1, random_state=0)

#create new a XBC model
print("XBC for click_bool...")
# eval_set = [(X_test,Y_test)]
# eval_metric = ["error"]
# classifier_XBC = XGBClassifier(booster = 'gbtree',learning_rate =0.001,
#                                n_estimators=5000,max_depth=8,gamma=0.2,
#                                use_label_encoder=False)
# classifier_XBC.fit(X_train, Y_train,early_stopping_rounds=20,eval_metric=eval_metric,eval_set=eval_set)
#
# print("Saving model to disk...")
# filename = 'finalized_model_XBC_click_bool.sav'
# pickle.dump(classifier_XBC, open(filename, 'wb'))
# print("Model saved!")

print("Loading model...")
filename = 'finalized_model_XBC_click_bool.sav'
classifier_XBC = pickle.load(open(filename, 'rb'))

yhat = classifier_XBC.predict(X_test)
accuracy = accuracy_score(Y_test,yhat)
print(accuracy)

#Predict booking bool
print("XBC for booking_bool...")

X_train2 = X_train2.assign(click_bool = classifier_XBC.predict(X_train))
X_test2 = X_test2.assign(click_bool = classifier_XBC.predict(X_test))
print(X_train2)
print(len(X_train2.columns))
eval_set = [(X_test2,Y_test)]
eval_metric = ["error"]
classifier_XBC = XGBClassifier(booster = 'gbtree',learning_rate =0.001,
                               n_estimators=5000,max_depth=4,gamma=0.2,
                               use_label_encoder=False)
classifier_XBC.fit(X_train2, Y_train,early_stopping_rounds=50,eval_metric=eval_metric,eval_set=eval_set)


print("Saving model to disk...")
filename = 'finalized_model_XBC_book_bool.sav'
pickle.dump(classifier_XBC, open(filename, 'wb'))
print("Model saved!")

yhat = classifier_XBC.predict(X_test2)
accuracy = accuracy_score(Y_test,yhat)
print(accuracy)
