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
import pickle


#Load data
print("Reading data")
data = pd.read_csv("feature_engineering.csv")
print(data.head(5))

data = data.drop(columns=["prob_booked","prob_clicked","date_time","position"])
data.replace([np.inf, -np.inf], int(0), inplace=True)
print(data.columns)
print(len(data.columns))

#normalize data
min_max_scaler = MinMaxScaler()
X,y = data.drop(["click_bool","booking_bool"],axis=1), data.loc[:,["click_bool","booking_bool"]]
X[['price_usd',"price_difference_user"]] = min_max_scaler.fit_transform(X[['price_usd',"price_difference_user"]])

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
# X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print("Started running model...")
forest = RandomForestClassifier(n_estimators=100, random_state=1)
classifier = MultiOutputClassifier(forest, n_jobs=-1)
classifier.fit(X_train, Y_train)
print(classifier.predict(X_test))
score = classifier.score(X_test,Y_test)
print(score)

# save the model to disk
print("Saving model to disk...")
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
print("Model saved!")

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
