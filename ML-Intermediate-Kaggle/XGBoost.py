import pandas as pd
from sklearn.model_selection import train_test_split

# Read data
data = pd.read_csv('melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

from xgboost import XGBRegressor

# n_jobs -> Cores to use for Parallelism
# early_stopping_rounds -> Model stops iterating after 5 consecutive deteriorations in validation scores

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(X_train, y_train, xgbregressor__early_stopping_rounds=5, xgbregressor__eval_set=[(X_valid, y_valid)], xgbregressor__verbose=True)

from sklearn.metrics import mean_absolute_error

preds = model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(preds, y_valid)))