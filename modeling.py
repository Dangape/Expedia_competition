import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import xgboost
from numpy import mean
from numpy import absolute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pickle
import time
import seaborn as sns
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
eval_set = [(X_test,Y_test)]
XBC = XGBRegressor(booster = 'gbtree',learning_rate =0.01,
 n_estimators=5000,max_depth=6,gamma=0.1,eval_metric="rmse",
                   early_stopping_rounds = 50,eval_set=eval_set,verbose=True)
classifier_XBC = MultiOutputRegressor(XBC)
classifier_XBC.fit(X_train, Y_train)

print("Saving model to disk...")
filename = 'finalized_model_XBC.sav'
pickle.dump(classifier_XBC, open(filename, 'wb'))
print("Model saved!")

# print("Loading model...")
# filename = 'finalized_model_XBC.sav'
# classifier_XBC = pickle.load(open(filename, 'rb'))

# evaluate model
y_train_pred1 = classifier_XBC.predict(X_train)
y_pred1 = classifier_XBC.predict(X_test)


train_mse1 = mean_squared_error(y_train_pred1, Y_train)
test_mse1 = mean_squared_error(y_pred1, Y_test)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)

print("Importance")
feature_importances = pd.DataFrame(classifier_XBC.estimators_[0].feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance")
plt.savefig("importance.png")
plt.show()

# imp = OrderedDict(sorted(classifier_XBC.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
# print("Importance:",imp)

print("%s minutes" %((time.time() - start_time)/60))