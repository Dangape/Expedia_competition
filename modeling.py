import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
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
# print(X.columns)
# print(len(X.columns))

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, y, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

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

#Create Neural Network model

print("Deep learning...")
model = Sequential()
model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='linear', kernel_initializer = 'normal'))
model.add(Dense(60, activation='linear', kernel_initializer = 'normal'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='linear', kernel_initializer = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='linear', kernel_initializer = 'normal'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='linear', kernel_initializer = 'normal'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid', kernel_initializer= 'normal'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['mean_absolute_error'])
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [es]
print(model.summary())
#fit the keras model on the dataset
hist = model.fit(X_train, Y_train,batch_size=10, epochs=50,validation_data=(X_val, Y_val),callbacks=callbacks_list,verbose=1)
Y_predicted = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_predicted)
rmse = np.sqrt(mse)

print('MSE : ',mse)
print('RMSE : ', rmse)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig("loss_model.png")
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#create new a XBC model
print("XBC...")
eval_set = [(X_test,Y_test)]
XBC = XGBRegressor(booster = 'gbtree',learning_rate =0.01,
 n_estimators=5000,max_depth=4,gamma=0.2,eval_metric="rmse",
                   early_stopping_rounds = 50,eval_set=eval_set,verbose=True)
classifier_XBC = MultiOutputRegressor(XBC)
classifier_XBC.fit(X_train, Y_train)

print("Saving model to disk...")
filename = 'finalized_model_XBC.sav'
pickle.dump(classifier_XBC, open(filename, 'wb'))
print("Model saved!")

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
plt.title("Feature importance XGBoost",fontsize=45)
plt.xlabel("Importance",fontsize=35)
plt.ylabel("Features",fontsize=35)
plt.savefig("importance_XBC.png")
plt.show()

print("%s minutes" %((time.time() - start_time)/60))