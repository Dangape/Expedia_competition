import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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

# load the model from disk
print("Loading model...")
filename = 'finalized_model_XBC.sav'
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

#Create submission file
print(type(loaded_pred))
d1 = {"predictions":loaded_pred.tolist()}
df = pd.DataFrame(d1)
df[["prob_clicked","prob_booked"]] = pd.DataFrame(df.predictions.tolist(), index = df.index)
df = df.drop(columns=["predictions"])

submission = pd.DataFrame({"srch_id":data["srch_id"],"prop_id":data["prop_id"],"prob_clicked":df["prob_clicked"],"prob_booked":df["prob_booked"]})
print(submission)

def relevance_grade(row):
    value = row["prob_clicked"]*1 + row["prob_booked"]*5
    # if ((row["click_bool"] == 0) & (row["booking_bool"] == 0)):
    #     return 0
    # if ((row["click_bool"] == 1) & (row["booking_bool"] == 0)):
    #     return 1
    # if ((row["click_bool"] == 0) & (row["booking_bool"] == 1)):
    #     return 5
    # if ((row["click_bool"] == 1) & (row["booking_bool"] == 1)):
    #     return 6
    return value

print("Generating relevance grade...")
submission["expected_relevance_grade"] = submission.apply(lambda row: relevance_grade(row), axis=1)
print(submission.head(10))
# print(submission.click_bool.unique())
# print(submission.booking_bool.unique())
submission.to_csv("submission.csv",index=False)

#Load submission
submission = pd.read_csv("submission.csv")

submission = submission.sort_values(by=['srch_id','expected_relevance_grade'], ascending=[True,False])
print(submission)
submission_final = submission.loc[:,['srch_id','prop_id']]
submission_final.to_csv("send.csv", index=False)


# unique_prop = submission.prop_id.unique() #unique property IDs
#
# for prop in tqdm(unique_prop):
#     appearences = len(submission[submission.prop_id == prop]) #count appearences of a prop id
#     booked = len(submission[(submission.prop_id==prop)&(submission.booking_bool==int(1))]) #count how many times a prop_id was booked
#     clicked = len(submission[(submission.prop_id==prop)&(submission.click_bool==int(1))]) #count how many time a prop_id was clicked
#     submission.loc[submission.prop_id==prop,"prob_booked"] = booked/appearences #probability of being booked
#     submission.loc[submission.prop_id == prop, "prob_clicked"] = clicked/appearences #probability of being clicked
# print(submission)
#
# submission.to_csv("submission.csv",index=False)