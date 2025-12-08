import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data in
data_path: str = 'data/melb_data.csv'
melb_data: pd.DataFrame = pd.read_csv(data_path)

# Target and Features
y: pd.Series = melb_data.Price
features: list[str] = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X: pd.DataFrame = melb_data[features]

# Split the data up
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Helper Function
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) -> float:
    model: DecisionTreeRegressor = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    prediction: np.ndarray = model.predict(val_X)
    mae: float = mean_absolute_error(val_y, prediction)
    return mae

# Test the MAE at different tree depths
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max Leaf Nodes: {max_leaf_nodes}   \t\t Mean Absolute Error: {mae}")

"""
Key Takeaways:
    - OVERFITTING: capturing spurious patterns that won't recur in the future, leading to less accurate results.
    - UNDERFITTING: failing to capture relevant data patterns, again leading to a loss of less accurate models
"""