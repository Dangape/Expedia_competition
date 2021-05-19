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
# print(len(X.columns))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#create new a XBC model
print("XBC...")
eval_set = [(X_test,Y_test)]
eval_metric = ["auc"]
model = XGBClassifier(booster = 'gbtree',learning_rate =0.001,
                               n_estimators=5000,max_depth=8,gamma=0.2,
                               use_label_encoder=False,objective="rank:ndcg",eval_metric=eval_metric)
classifier_XBC = MultiOutputClassifier(model)
classifier_XBC.fit(X_train, Y_train)

print("Saving model to disk...")
filename = 'finalized_model_XBC_bool.sav'
pickle.dump(classifier_XBC, open(filename, 'wb'))
print("Model saved!")

#evaluate model
yhat = classifier_XBC.predict(X_test)
auc_y1 = roc_auc_score(Y_test.iloc[:,0],yhat[:,0])
auc_y2 = roc_auc_score(Y_test.iloc[:,1],yhat[:,1])

print("Classifier score: %.4f" % classifier_XBC.score(X_test,Y_test))
print("ROC AUC y1: %.4f, y2: %.4f" % (auc_y1, auc_y2))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test.iloc[:,0], yhat[:,0])
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(Y_test.iloc[:,1], yhat[:,1])

cr_y1 = classification_report(Y_test.iloc[:,0],yhat[:,0])
cr_y2 = classification_report(Y_test.iloc[:,1],yhat[:,1])
print(cr_y1)
print(cr_y2)

print("Importance XBC...")
feature_importances = pd.DataFrame(classifier_XBC.estimators_[0].feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance XGBoost",fontsize=45)
plt.xlabel("Importance",fontsize=35)
plt.xticks(fontsize=16)
plt.ylabel("Features",fontsize=35)
plt.yticks(fontsize=15)
plt.savefig("importance_XBC_click.png")
plt.show()

print("%s minutes" %((time.time() - start_time)/60))