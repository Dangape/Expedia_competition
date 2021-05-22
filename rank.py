import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error

start_time = time.time()
#Load data
print("Reading data...")
data = pd.read_csv("feature_engineering.csv")
print(data.head(5))

data = data.drop(columns=["prob_booked","prob_clicked","date_time","position","click_bool","booking_bool"])
data.replace([np.inf, -np.inf], int(0), inplace=True)

#normalize data
min_max_scaler = MinMaxScaler()
data[['price_usd',"price_difference_user"]] = min_max_scaler.fit_transform(data[['price_usd',"price_difference_user"]])

# # load the model from disk
print("Loading model...")
filename = 'finalized_model_XBC_bool.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prediction
print("Loaded! Now, predicting probas...")
pred = loaded_model.predict_proba(data)
print(pred)

df1 = pd.DataFrame(pred[0],columns=["prob_clicked_0","prob_clicked_1"])
df2 = pd.DataFrame(pred[1],columns=["prob_booked_0","prob_booked_1"])
df3 = pd.DataFrame({"prob_clicked":df1["prob_clicked_1"],"prob_booked":df2["prob_booked_1"]})

new_data = pd.concat([data,df3],axis=1)
print(new_data.head(5))
print(new_data.columns)

def relevance_grade(row):
    value = row["prob_clicked"]*1 + row["prob_booked"]
    return value

print("Generating relevance grade...")
new_data["expected_relevance_grade"] = new_data.apply(lambda row: relevance_grade(row), axis=1)
print(new_data.head(10))

X,y = new_data.drop(["expected_relevance_grade"],axis=1), new_data.loc[:,["expected_relevance_grade"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#XHBoostRegressor Model
print("Running XGBRegressor")
# eval_set = [(X_test,Y_test)]
# eval_metric = ["ndcg"]
model = XGBRegressor(booster = 'gbtree',learning_rate =0.01,
                               n_estimators=3000,max_depth=4,gamma=0.2,
                               use_label_encoder=False,objective="rank:ndcg")
model.fit(X_train, Y_train)
yhat = model.predict(X_test)
rms = mean_squared_error(Y_test, yhat, squared=False)

#
print("Saving model to disk...")
filename = 'finalized_model_XBR_rank.sav'
pickle.dump(model, open(filename, 'wb'))
print("Model saved!")

print("Importance XBR...")
feature_importances = pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance')
print(feature_importances)
figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
sns.barplot(x= feature_importances.importance,y =feature_importances.index)
plt.title("Feature importance XGBoost Rank",fontsize=45)
plt.xlabel("Importance",fontsize=35)
plt.xticks(fontsize=10)
plt.ylabel("Features",fontsize=35)
plt.yticks(fontsize=10)
plt.savefig("importance_XBR_rank.png")
plt.show()

print("%s minutes" %((time.time() - start_time)/60))