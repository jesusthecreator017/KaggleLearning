import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data in
data_path: str = "data/melb_data.csv"
melb_data: pd.DataFrame = pd.read_csv(data_path)

# Filter the empty cells out
melb_data = melb_data.dropna(axis=0)

# Target and Features
y: pd.Series = melb_data.Price
features: list[str] = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X: pd.DataFrame = melb_data[features]

# Split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Create and fit the model
forest_model: RandomForestRegressor = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds: np.ndarray = forest_model.predict(val_X)
print(f"Mean Absolute Error: {mean_absolute_error(val_y, melb_preds)}")