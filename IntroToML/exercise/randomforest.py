import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data in
data_path: str = 'data/train.csv'
home_data: pd.DataFrame = pd.read_csv(data_path)

# Target and Features
y: pd.Series = home_data.SalePrice
features: list[str] = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X: pd.DataFrame = home_data[features]

# Split the data up
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

home_rf_model: RandomForestRegressor = RandomForestRegressor(random_state=1)
home_rf_model.fit(train_X, train_y)
y_pred: np.ndarray = home_rf_model.predict(val_X)
mae = mean_absolute_error(val_y, y_pred)
print(f"Random Forest MAE: {mae}")