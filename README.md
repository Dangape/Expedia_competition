# Expedia_competition
The intention of this project is to study the solutions made for the original Personalize Expedia Hotel Searches that occurred at the International Conference on Data Mining\
(ICDM) in 2013. The Kaggle competition link can be found [here](https://www.kaggle.com/c/expedia-personalized-sort/data).

Furthermore, there is a Dropbox link [here](https://www.dropbox.com/sh/5kedakjizgrog0y/_LE_DFCA7J/ICDM_2013) where you can find the top 3 solutions with the best score.

# Files
- The `cleaning_data.py` file contains the main code for cleaning the data and creating new features (feature engineering)
- Since the code takes too long to run (aprox. 10h), the `feature_engineering.zip` file contains the cleaned data set in  a `.pkl` format
- In order to open the cleaned file in python use the `pandas` built-in function `pd.read_pickle`
