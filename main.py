import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load in the Housing Data
data_path = "melb_data.csv"
melb_data: pd.DataFrame = pd.read_csv(data_path)

# Take a peek at the data
print(f"Quick Peek:\n {melb_data.head()}")

# Look for our possible featurs
features = [feat for feat in melb_data.columns]
print(f"Features:\n{features}")

# Select our Prediction Target
y = melb_data.Price


# Select our features
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[features]

# Build the model
melb_model = DecisionTreeRegressor()
melb_model.fit(X, y)

# Make predictions
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melb_model.predict(X.head()))git remote set-url origin git@github.com:jesusthecreator017/KaggleLearning.git
