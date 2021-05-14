import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import time

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
print(X.columns)
print(len(X.columns))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
# X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print("Started running model...")
#Create Random forest model
print("Random Forest...")
forest = RandomForestClassifier(n_estimators=200, random_state=0)
classifier_rf = MultiOutputClassifier(forest, n_jobs=-1)
classifier_rf.fit(X_train, Y_train)
score = classifier_rf.score(X_test,Y_test)
print(score)
#create new a knn model
print("Knn...")
knn = KNeighborsClassifier(n_neighbors=10)
classifier_knn = MultiOutputClassifier(knn, n_jobs=-1)
classifier_knn.fit(X_train, Y_train)
score = classifier_knn.score(X_test,Y_test)
print(score)
#create a new logistic regression model
print("Logistic regression...")
log_reg = LogisticRegression()
classifier_log_reg = MultiOutputClassifier(log_reg, n_jobs=-1)
classifier_log_reg.fit(X_train, Y_train)
score = classifier_log_reg.score(X_test,Y_test)
print(score)

# classifier = MultiOutputClassifier(forest, n_jobs=-1)
# classifier.fit(X_train, Y_train)
# print(classifier.predict(X_test))
# score = classifier.score(X_test,Y_test)
# print(score)

#create a dictionary of our models
print("Ensembling...")
estimators=[("knn", classifier_knn), ("rf", classifier_rf),("log_reg",classifier_log_reg)]#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting="hard")
classfier_ensembled = MultiOutputClassifier(ensemble, n_jobs=-1)
classfier_ensembled.fit(X_train, Y_train)#test our model on the test data
print(classfier_ensembled.score(X_test, Y_test))


# save the model to disk
print("Saving model to disk...")
filename = 'finalized_model_ensembled.sav'
pickle.dump(classfier_ensembled, open(filename, 'wb'))
print("Model saved!")

# load the model from disk
print("Loading model...")
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

print("%s seconds" %(time.time() - start_time))