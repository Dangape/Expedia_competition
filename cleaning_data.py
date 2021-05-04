import pandas as pd
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

#load data
data = pd.read_csv("Data/training_set_VU_DM.csv")
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

#Creating new features
data = data.drop(columns=["gross_booking_usd"]) #not an interesting feature
data["price_difference_history"] = data["prop_log_historical_price"] - np.log(data["price_usd"])
data["price_difference_user"] = data["visitor_hist_adr_usd"] - data["price_usd"]
data["starrating_diff"] = data["visitor_hist_starrating"] - data["prop_starrating"]
data = data.drop(columns=["visitor_hist_starrating"]) #drop after calculating starrating_diff

#Filling NAs
data["srch_query_affinity_score"] = data["srch_query_affinity_score"].fillna(min(data["srch_query_affinity_score"]))

for prop in tqdm(unique_prop):
    data.loc[data.prop_id == prop,"prop_review_score"] = data.loc[data.prop_id == prop,"prop_review_score"].fillna(min(data.loc[data.prop_id==prop,"prop_review_score"]))
    data.loc[data.prop_id == prop, "prop_location_score2"] = data.loc[data.prop_id == prop, "prop_location_score2"].fillna(min(data.loc[data.prop_id == prop, "prop_location_score2"]))

unique_srch_dest = data.srch_destination_id.unique()
unique_visitor_loc = data.visitor_location_country_id.unique()

for destination in tqdm(unique_srch_dest):
    for visitor in unique_visitor_loc:
        data.loc[(data.srch_destination_id==destination)&(data.visitor_location_country_id==visitor),"orig_destination_distance"] = \
            data.loc[(data.srch_destination_id==destination)&(data.visitor_location_country_id==visitor),"orig_destination_distance"].\
                fillna(data.loc[(data.srch_destination_id==destination)&(data.visitor_location_country_id==visitor),"orig_destination_distance"].mean())

data["orig_destination_distance"] = data["orig_destination_distance"].fillna(data["orig_destination_distance"].mean()) #replace remaining NAs with mean

#Saving new data frame
print("Saving file to disk...")
data.to_pickle("feature_engineering.pkl") #save dataframe with new features
print("File saved!")

#Load saved dataframe
print("Loading data...")
data = pd.read_pickle("feature_engineering.pkl")
print("Loaded!")
print(data)+