import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import pickle
import time
import seaborn as sns

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

#Create Ridge Regressor model
print("Ridge model...")
regressor = Ridge(alpha=100)
classifier_ridge = MultiOutputRegressor(regressor)
classifier_ridge.fit(X_train, Y_train)

y_train_pred1 = classifier_ridge.predict(X_train)
y_pred1 = classifier_ridge.predict(X_test)
train_mse1 = mean_squared_error(y_train_pred1, Y_train)
test_mse1 = mean_squared_error(y_pred1, Y_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)

print("Saving model to disk...")
filename = 'finalized_model_ridge.sav'
pickle.dump(classifier_ridge, open(filename, 'wb'))
print("Model saved!")

#Create SGD Regressor model
print("SGD Regressor...")
regressor = SGDRegressor(eta0=0.01,epsilon=0.0001,loss='squared_epsilon_insensitive',learning_rate='adaptive')
classifier_SGD = MultiOutputRegressor(regressor)
classifier_SGD.fit(X_train, Y_train)

y_train_pred1 = classifier_SGD.predict(X_train)
y_pred1 = classifier_SGD.predict(X_test)
train_mse1 = mean_squared_error(y_train_pred1, Y_train)
test_mse1 = mean_squared_error(y_pred1, Y_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)

print("Saving model to disk...")
filename = 'finalized_model_SGD.sav'
pickle.dump(classifier_SGD, open(filename, 'wb'))
print("Model saved!")

#Create SVR model
print("SVR...")
regressor = SVR(epsilon=0.2)
classifier_SVR = MultiOutputRegressor(regressor)
classifier_SVR.fit(X_train, Y_train)

y_train_pred1 = classifier_SVR.predict(X_train)
y_pred1 = classifier_SVR.predict(X_test)
train_mse1 = mean_squared_error(y_train_pred1, Y_train)
test_mse1 = mean_squared_error(y_pred1, Y_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)

print("Saving model to disk...")
filename = 'finalized_model_SVR.sav'
pickle.dump(classifier_SVR, open(filename, 'wb'))
print("Model saved!")

#create new a XBC model
print("XBC...")
# eval_set = [(X_test,Y_test)]
# XBC = XGBRegressor(booster = 'gbtree',learning_rate =0.01,
#  n_estimators=5000,max_depth=4,gamma=0.2,eval_metric="rmse",
#                    early_stopping_rounds = 50,eval_set=eval_set,verbose=True)
# classifier_XBC = MultiOutputRegressor(XBC)
# classifier_XBC.fit(X_train, Y_train)
#
# print("Saving model to disk...")
# filename = 'finalized_model_XBC.sav'
# pickle.dump(classifier_XBC, open(filename, 'wb'))
# print("Model saved!")
# load the model from disk
print("Loading model...")
filename = 'finalized_model_XBC.sav'
classifier_XBC = pickle.load(open(filename, 'rb'))
# evaluate model
y_train_pred1 = classifier_XBC.predict(X_train)
y_pred1 = classifier_XBC.predict(X_test)
train_mse1 = mean_squared_error(y_train_pred1, Y_train)
test_mse1 = mean_squared_error(y_pred1, Y_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)

print("Importance XBC...")
feature_importances = pd.DataFrame(classifier_XBC.estimators_[0].feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance XGBoost",fontsize=50)
plt.savefig("importance_XBC.png")
plt.show()

print("Importance Ridge...")
feature_importances = pd.DataFrame(classifier_ridge.estimators_[0].feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance Ridge",fontsize=50)
plt.savefig("importance_ridge.png")
plt.show()

print("Importance SGD..")
feature_importances = pd.DataFrame(classifier_SGD.estimators_[0].feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance SGD",fontsize=50)
plt.savefig("importance_SGD.png")
plt.show()

print("Importance SVR...")
feature_importances = pd.DataFrame(classifier_SVR.estimators_[0].feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance SVR",fontsize=50)
plt.savefig("importance_SVR.png")
plt.show()
print("%s minutes" %((time.time() - start_time)/60))