import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

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

dropped_train_X = train_X.drop(cols_with_missing, axis=1)
dropped_val_X = val_X.drop(cols_with_missing, axis=1)

print(f"MAE Approach #1 (Drop Columns with Missing Values): {score_dataset(dropped_train_X, dropped_val_X, train_y, val_y)}")

# Approach #2: Imputation
my_imputer = SimpleImputer()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_val_X = my_imputer.transform(val_X)

print(f"MAE Approach #2 (Impute Into Missing Values): {score_dataset(imputed_train_X, imputed_val_X, train_y, val_y)}")

# Appraoch #3: Extension to Imputing
X_train_plus = train_X.copy()
X_val_plus = val_X.copy()

for col in cols_with_missing:
    X_train_plus[col + "_was_missing"] = X_train_plus[col].isnull()
    X_val_plus[col + "_was_missing"] = X_val_plus[col].isnull()

my_new_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_new_imputer.fit_transform(X_train_plus))
imputed_X_val_plus = pd.DataFrame(my_new_imputer.transform(X_val_plus))

print(f"MAE Approach #3 (An Extension to Imputing): {score_dataset(imputed_X_train_plus, imputed_X_val_plus, train_y, val_y)}")