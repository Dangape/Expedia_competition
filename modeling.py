import pandas as pd
import numpy as np

data = pd.read_csv("feature_engineering.csv")
print(data)

#Show missing values percentege for each column
for column in data.columns:
    percent = (data[str(column)].isna().sum())/(len(data[str(column)]))*100
    print(str(column),":",float(percent))
