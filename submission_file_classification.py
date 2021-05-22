import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time


start_time = time.time()
#Loading test data
print("Reading data...")
data = pd.read_csv("feature_engineering_test.csv")
print(data.head(5))
print(data.columns)
print(len(data.columns))

data = data.drop(columns=["date_time"])
data.replace([np.inf, -np.inf], int(0), inplace=True)
min_max_scaler = MinMaxScaler()
data[['price_usd',"price_difference_user"]] = min_max_scaler.fit_transform(data[['price_usd',"price_difference_user"]])

# # load the model from disk
print("Loading model...")
filename = 'finalized_model_XBC_bool.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prediction
print("Loaded! Now, predicting...")
pred = loaded_model.predict_proba(data)
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

# print(loaded_pred)
# print(len(loaded_pred))

#Create submission file
# print(type(loaded_pred))
# print(loaded_pred[0][1][1])

# d1 = {"prob_clicked":loaded_pred[0],"prob_booked":loaded_pred[1]}
df1 = pd.DataFrame(loaded_pred[0],columns=["prob_clicked_0","prob_clicked_1"])
df2 = pd.DataFrame(loaded_pred[1],columns=["prob_booked_0","prob_booked_1"])
df3 = pd.DataFrame({"prob_clicked":df1["prob_clicked_1"],"prob_booked":df2["prob_booked_1"]})

print(df3)
new_data = pd.concat([data,df3],axis=1)
print(new_data.head(5))

#Load pred rank
filename = 'finalized_model_XBR_rank.sav'
loaded_model = pickle.load(open(filename, 'rb'))
rank = loaded_model.predict(new_data)
print(rank)

# submission = pd.DataFrame({"srch_id":data["srch_id"],"prop_id":data["prop_id"],"prob_clicked":df3["prob_clicked"],"prob_booked":df3["prob_booked"]})
# print(submission)
#
submission =  pd.DataFrame({"srch_id":data["srch_id"],"prop_id":data["prop_id"],"expected_relevance_grade":rank})
print(submission)
# def relevance_grade(row):
#     value = row["prob_clicked"]*1 + row["prob_booked"]
#     return value


print("Generating relevance grade...")
# submission["expected_relevance_grade"] = submission.apply(lambda row: relevance_grade(row), axis=1)
# print(submission.head(10))

submission.to_csv("submission.csv",index=False)

#Load submission
submission = pd.read_csv("submission.csv")

submission = submission.sort_values(by=['srch_id','expected_relevance_grade'], ascending=[True,False])
print(submission)
submission_final = submission.loc[:,['srch_id','prop_id']]
submission_final.to_csv("send.csv", index=False)
print("Done! You can submit the file!")
print("%s minutes" %((time.time() - start_time)/60))