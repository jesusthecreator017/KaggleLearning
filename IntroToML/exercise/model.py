import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Load in the data
data_path: str = 'data/train.csv'
home_data: pd.DataFrame = pd.read_csv(data_path)

# Chose a Prediction Target
y: pd.Series = home_data.SalePrice

# Choose the Features
features: list[str] = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X: pd.DataFrame = home_data[features]

# Review the data
print(X.describe())
print(X.head())

# Develop the model
home_model: DecisionTreeRegressor = DecisionTreeRegressor()
home_model.fit(X, y)

# Make predictions
predictions: np.ndarray = home_model.predict(X.head())
print(f"Prediction vs Actual:\n{predictions}\n{y[:5]}")