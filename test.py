import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from numba import njit

#Load saved dataframe
print("Loading data...")
data = pd.read_pickle("feature_engineering.pkl")
print("Loaded!")
print(data.head())
print(data.columns)

#data["srch_query_affinity_score"] = data["srch_query_affinity_score"].fillna(data["srch_query_affinity_score"].min())
print(data["prop_review_score"].unique)

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