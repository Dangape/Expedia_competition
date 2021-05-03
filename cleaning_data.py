import pandas as pd
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt

#load data
data = pd.read_csv("Data/training_set_VU_DM.csv")
print(data.head())
print(data.columns)

#Plot missing values
for column in data.columns:
    percent = (data[str(column)].isna().sum())/(len(data[str(column)]))*100
    print(str(column),":",float(percent))

#Creating new features
data["price_difference_history"] = data["prop_log_historical_price"] - np.log(data["price_usd"])
data["price_difference_user"] = data["visitor_his_adr_usd"] - data["price_usd"]

for prop in data["prop_id"].unique():
    appearences = len(data[data.prop_id == prop])
    booked = data[(data.prop_id==prop)&(data.booking_bool==int(1))]
    data[data.prop_id==prop]["prob_booked"] = booked/appearences

