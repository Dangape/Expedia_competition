import pandas as pd
import pickle
import numpy as np

#Loading test data
print("Reading data...")
data = pd.read_csv("feature_engineering_test.csv")
print(data.head(5))
print(data.columns)
print(len(data.columns))

data = data.drop(columns=["date_time","random_bool"])
data.replace([np.inf, -np.inf], int(0), inplace=True)

# load the model from disk
print("Loading model...")
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prediction
print("Loaded! Now, predicting...")
pred = loaded_model.predict(data)
print(pred)

#Save prediction
print("Saving prediction...")
open_file = open("pred.pkl","wb")
pickle.dump(pred,open_file)
open_file.close()

#Load prediction
print("Loading file...")
open_file = open("pred.pkl","rb")
loaded_pred = pickle.load(open_file)
open_file.close()
print(loaded_pred.unique())
