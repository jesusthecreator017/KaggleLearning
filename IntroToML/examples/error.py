import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data in
data_path: str = 'data/melb_data.csv'
melb_data: pd.DataFrame = pd.read_csv(data_path)

# Filter rows with missing price values
filtered_melb_data: pd.DataFrame = melb_data.dropna(axis=0)

# Choose a prediction target and features
y: pd.Series = filtered_melb_data.Price
features: list[str] = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X: pd.DataFrame = filtered_melb_data[features]

# Define and Fit the model
melb_model = DecisionTreeRegressor()
melb_model.fit(X, y)

# Calculate the Error
predicted_home_prices: np.ndarray = melb_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)
print(f"Mean Absolute Error: {mae}")

# Get an accurate score by splitting the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

new_melb_model = DecisionTreeRegressor()
new_melb_model.fit(train_X, train_y)

val_predictions = new_melb_model.predict(val_X)
new_mae = mean_absolute_error(val_y, val_predictions)
print(f"New Mean Absolute Error: {new_mae}")