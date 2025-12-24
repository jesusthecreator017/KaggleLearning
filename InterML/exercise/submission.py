import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load in the data
train_data_dir = 'data/train.csv'
test_data_dir = 'data/test.csv'

X_full: pd.DataFrame = pd.read_csv(train_data_dir, index_col='Id')
X_test_full: pd.DataFrame = pd.read_csv(test_data_dir, index_col='Id')

# Target and Features
y: pd.Series = X_full.SalePrice
features: list[str] = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X: pd.DataFrame = X_full[features].copy()
X_test: pd.DataFrame = X_test_full[features].copy()

# Split the data up
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=.8, test_size=.2, random_state=0)

# Create Models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# Test the models performance
def score_model(model, t_X=train_X, v_X=val_X, t_y=train_y, v_y=val_y) -> float:
    model.fit(t_X, t_y)
    y_pred: np.ndarray = model.predict(v_X)
    return mean_absolute_error(v_y, y_pred)

for i in range(len(models)):
    score = score_model(models[i])
    print(f"Model #{i + 1}\t\tMAE: {score}")

# Model #3 is the best model utilize it for the competition
my_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
my_model.fit(X, y)

preds_test = my_model.predict(X_test)

output = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': preds_test
})

output.to_csv('data/submission.csv', index=False)

# Read in the submission
sub = pd.read_csv('data/submission.csv')

print(f"Shape of Submission: {sub.shape}")
print(f"Last few entries of Submission:\n{sub.loc[1450:]}")
print("daily commit")
print("Daily commit 2")
print("Daily commit 3")
print("daily Commit 4")