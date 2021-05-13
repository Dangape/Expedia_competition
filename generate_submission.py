import pandas as pd
import pickle

#Loading test data
print("Reading data...")
data = pd.read_csv("feature_engineering_test.csv")
print(data.columns)
print(data.isnull().values.any())

print(data.columns)
print(len(data.columns))


is_NaN = data.isnull()

row_has_NaN = is_NaN.any(axis=1)

rows_with_NaN = data[row_has_NaN]


print(rows_with_NaN)
# #Show missing values percentege for each column
# for column in data.columns:
#     percent = (data[str(column)].isna().sum())/(len(data[str(column)]))*100
#     print(str(column),":",float(percent))

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
