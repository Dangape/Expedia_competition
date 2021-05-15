import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost
from numpy import mean
from numpy import absolute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import pickle
import time
from collections import OrderedDict

start_time = time.time()
#Load data
print("Reading data...")
data = pd.read_csv("feature_engineering.csv")
print(data.head(5))

data = data.drop(columns=["click_bool","booking_bool","date_time","position"])
data.replace([np.inf, -np.inf], int(0), inplace=True)

#normalize data
min_max_scaler = MinMaxScaler()
X,y = data.drop(["prob_booked","prob_clicked"],axis=1), data.loc[:,["prob_booked","prob_clicked"]]
X[['price_usd',"price_difference_user"]] = min_max_scaler.fit_transform(X[['price_usd',"price_difference_user"]])
print(X.columns)
print(len(X.columns))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
#X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print("Started running model...")
# #Create Random forest model
# print("Random Forest...")
# forest = RandomForestClassifier(n_estimators=200, random_state=0)
# classifier_rf = MultiOutputClassifier(forest)
# classifier_rf.fit(X_train, Y_train)
# score = classifier_rf.score(X_test,Y_test)
# print(score)
#
# print("Saving model to disk...")
# filename = 'finalized_model_rf.sav'
# pickle.dump(classifier_rf, open(filename, 'wb'))
# print("Model saved!")

#create new a XBC model
print("XBC...")
XBC = XGBRegressor(booster = 'gbtree',learning_rate =0.01,
 n_estimators=5000,early_stopping_rounds=50,evals=[X_test,Y_test],max_depth=5,gamma=0.1,eval_metric="rmse")
classifier_XBC = MultiOutputRegressor(XBC)
classifier_XBC.fit(X_train, Y_train)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(classifier_XBC, X_train, Y_train, scoring='neg_root_mean_squared_error', cv=cv)
# force scores to be positive
scores = absolute(scores)
print(scores)
print('Mean RMSE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

print("Saving model to disk...")
filename = 'finalized_model_XBC.sav'
pickle.dump(classifier_XBC, open(filename, 'wb'))
print("Model saved!")


print("%s seconds" %(time.time() - start_time))

xgboost.plot_importance(classifier_XBC)
plt.savefig("importance")
plt.show()

imp = OrderedDict(sorted(classifier_XBC.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
print("Importance:",imp)