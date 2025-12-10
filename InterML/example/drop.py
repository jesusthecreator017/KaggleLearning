import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
data_dir: str = ('data/melb_data.csv')
melb_data: pd.DataFrame = pd.read_csv(data_dir)

# Target and Features (Only include numerical predictors)
y = melb_data.Price
melb_predictors: pd.DataFrame = melb_data.drop('Price', axis=1)
X: pd.DataFrame = melb_predictors.select_dtypes(exclude=['object'])

# Split up the data
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Function to score the different approaches
def score_dataset(X_train, X_val, y_train, y_val):
    model: RandomForestRegressor = RandomForestRegressor(n_estimators=10 ,random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_absolute_error(y_pred, y_val)

# Approach 1: Drop Columns with missing values
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

dropped_train_X = train_X.drop(cols_with_missing)
dropped_val_X = val_X.drop(cols_with_missing)
