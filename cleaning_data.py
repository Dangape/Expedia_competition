import pandas as pd
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# #load data
data = pd.read_csv("Data/training_set_VU_DM.csv") #training data
print(data.head())
print(data.columns)

#Show missing values percentege for each column
for column in data.columns:
    percent = (data[str(column)].isna().sum())/(len(data[str(column)]))*100
    print(str(column),":",float(percent))

unique_prop = data.prop_id.unique() #unique property IDs

#Run this part just if you need to change the code, since it takes too long to finish the loop
for prop in tqdm(unique_prop):
    appearences = len(data[data.prop_id == prop]) #count appearences of a prop id
    booked = len(data[(data.prop_id==prop)&(data.booking_bool==int(1))]) #count how many times a prop_id was booked
    clicked = len(data[(data.prop_id==prop)&(data.click_bool==int(1))]) #count how many time a prop_id was clicked
    data.loc[data.prop_id==prop,"prob_booked"] = booked/appearences #probability of being booked
    data.loc[data.prop_id == prop, "prob_clicked"] = clicked/appearences #probability of being clicked

unique_prop = data.prop_id.unique() #unique property IDs

# #Creating new features
data = data.drop(columns=["gross_bookings_usd"]) #not an interesting feature
data["price_difference_history"] = data["prop_log_historical_price"] - np.log(data["price_usd"])
data["visitor_hist_adr_usd"] = data["visitor_hist_adr_usd"].fillna(0)
data["price_difference_user"] = abs(data["visitor_hist_adr_usd"] - data["price_usd"])
data["visitor_hist_starrating"] = data["visitor_hist_starrating"].fillna(0)
data["starrating_diff"] = abs(data["visitor_hist_starrating"] - data["prop_starrating"])
data = data.drop(columns=["visitor_hist_starrating"]) #drop after calculating starrating_diff

#Filling NAs
data["srch_query_affinity_score"] = data["srch_query_affinity_score"].fillna(data["srch_query_affinity_score"].min())

for prop in tqdm(unique_prop):
    data.loc[data.prop_id == prop,"prop_review_score"] = data.loc[data.prop_id == prop,"prop_review_score"].fillna(data.loc[data.prop_id==prop,"prop_review_score"].min())
    data.loc[data.prop_id == prop, "prop_location_score2"] = data.loc[data.prop_id == prop, "prop_location_score2"].fillna(data.loc[data.prop_id == prop, "prop_location_score2"].min())

for i in list(data.columns)[26:50]:
    data[str(i)] = data[str(i)].fillna(0) #replaceing competitors variables


# def replace_NaN(x):
#     rows_NaN = x[x["orig_destination_distance"].isnull()]
#     absolute_mean = x.orig_destination_distance.mean()
#     for ind in tqdm(range(0,len(rows_NaN))):
#         destination = x.loc[ind,"srch_destination_id"]
#         visitor = x.loc[ind,"visitor_location_country_id"]
#         filter = x.query('srch_destination_id==@destination & visitor_location_country_id== @visitor')
#         mean = x.query('srch_destination_id==@destination & visitor_location_country_id== @visitor').mean()
#         if len(filter)>1:
#             filter["orig_destination_distance"] = filter["orig_destination_distance"].fillna(mean)
#         else:
#             filter["orig_destination_distance"] = filter["orig_destination_distance"].fillna(absolute_mean)
#     return x
#
# data = replace_NaN(data)
# print(data)

data["orig_destination_distance"] = data["orig_destination_distance"].fillna(data["orig_destination_distance"].mean())
data["prop_review_score"] = data["prop_review_score"].fillna(data["prop_review_score"].mean()) #Replace ramaining NAs
data["prop_location_score2"] = data["prop_location_score2"].fillna(data["prop_location_score2"].mean()) #replacing remaining 3% NAs

for column in data.columns:
    percent = (data[str(column)].isna().sum())/(len(data[str(column)]))*100
    print(str(column),":",float(percent))
#
#Saving new data frame
print("Saving file to disk...")
data.to_pickle("feature_engineering.pkl") #save dataframe with new features
print("File saved!")
