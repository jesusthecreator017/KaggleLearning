import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load in the data
data_path: str = "data/train.csv"
home_data: pd.DataFrame = pd.read_csv(data_path)

# Target and Features
y: pd.Series = home_data.SalePrice
features: list[str] = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X: pd.DataFrame = home_data[features]

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Create and fit the model
model_one = DecisionTreeRegressor(random_state=1)
model_one.fit(train_X, train_y)

# Display predictions and mae
predictions = model_one.predict(val_X)
mae_one = mean_absolute_error(val_y, predictions)

print(f"Model One Predictions: {predictions[:5]}")
print(f"Model One MAE: {mae_one}\n")

# Helper MAE function
def get_mae(max_leaf_nodes, t_X, v_X, t_y, v_y) -> float:
    model: DecisionTreeRegressor = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(t_X, t_y)
    val_pred: np.ndarray = model.predict(v_X)
    mae: float = mean_absolute_error(v_y, val_pred)
    return mae

# Compare the model mae at different lead node counts
for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    curr_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max Leaf Nodes: {max_leaf_nodes}   \t\t\t MAE: {curr_mae}")

# Roughly at 100 leafs we have the lowest MAE
best_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=100)
best_model.fit(train_X, train_y)

predictions = best_model.predict(val_X)
mae_one = mean_absolute_error(val_y, predictions)
print(f"\nPredictions: {predictions[:5]}")
print(f"Model One MAE: {mae_one}\n")

