import pandas as pd
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle


#Load data
#Loading test data
print("Reading data...")
data = pd.read_csv("feature_engineering_test.csv")
print(data.head(5))
print(data.columns)
print(len(data.columns))

data = data.drop(columns=["date_time"])
data.replace([np.inf, -np.inf], int(0), inplace=True)

#Load models
#Ridge regression
print("Loading XGBoost...")
filename = 'finalized_model_ridge.sav'
classifier_XBC = pickle.load(open(filename, 'rb'))

#XGBoostRegressor
print("Loading XGBoost...")
filename = 'finalized_model_XBC.sav'
classifier_XBC = pickle.load(open(filename, 'rb'))

# Neural Network
print("Loading NN...")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
NN = model_from_json(loaded_model_json)
# load weights into new model
NN.load_weights("model.h5")
print("Loaded model from disk")

#Check if model was loaded correctly
Y_predicted = NN.predict(X_test)
mse = mean_squared_error(Y_test, Y_predicted)
mae = mean_absolute_error(Y_test, Y_predicted)
rmse = np.sqrt(mse)

print('MSE : ',mse)
print('MAE : ', mae)
print('RMSE : ', rmse)

#Stacking models