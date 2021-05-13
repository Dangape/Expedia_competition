# Expedia_competition
The intention of this project is to study the solutions made for the original Personalize Expedia Hotel Searches that occurred at the International Conference on Data Mining\
(ICDM) in 2013. The Kaggle competition link can be found [here](https://www.kaggle.com/c/expedia-personalized-sort/data).

Furthermore, there is a Dropbox link [here](https://www.dropbox.com/sh/5kedakjizgrog0y/_LE_DFCA7J/ICDM_2013) where you can find the top 3 solutions with the best score.

# Files
- The `cleaning_data_training.py` file contains the main code for cleaning the training data and creating new features (feature engineering)
- The `cleaning_data_test.py` file contains the main code for cleaning the test data and creating new features (feature engineering)
- Since the code takes too long to run (aprox. 8h), the `feature_engineering.rar` file contains the cleaned data set in  a `.csv` format
- The `modeling.py` file trains the model and save it as a `.sav` file
- The `generate_submission.py` creates a prediction for the test file and generates the file to submit on Kaggle


# Assigning relevance grades
- 5: the user purchased a room at this hotel
- 1: the user clicked to see more information about this hotel
- 0: the user neither clicked nor purchased this hotel