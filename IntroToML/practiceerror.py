import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data
data_path: str = 'data/train.csv'
home_data: pd.DataFrame = pd.read_csv(data_path)

# Prediction Target and Features
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Model Definition and Fitting
home_model = DecisionTreeRegressor()
home_model.fit(X, y)

# Get the in-place error
prediction = home_model.predict(X)
in_place_mae = mean_absolute_error(y, prediction)

# Print the results
print(f"First In-Sample prediction: {prediction}")
print(f"Actual Target Values: {y.head().tolist()}")

# Split the date
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.4)

# Create the new model on training data
new_home_model = DecisionTreeRegressor()
new_home_model.fit(train_X, train_y)

# Evaluate the model on the validation data
new_prediction = new_home_model.predict(val_X)
new_mae = mean_absolute_error(val_y, new_prediction)

print(f"In-Place MAE: {in_place_mae}")
print(f"Actual MAE: {new_mae}")