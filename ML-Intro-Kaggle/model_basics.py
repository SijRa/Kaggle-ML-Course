import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Descision Tree
from sklearn.tree import DecisionTreeRegressor
# RandomForest
from sklearn.ensemble import RandomForestRegressor


melbourne_data = pd.read_csv('melb_data.csv')
filtered_melbourne_data = melbourne_data.dropna(axis=0) # Drop missing values

# Target to predict
y = filtered_melbourne_data.Price

# Features to predict price
melbourne_features = [
    'Rooms', 
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude' 
    ]

# Extract features from data
X = filtered_melbourne_data[melbourne_features]

# Define model 
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X,y)

print("Making predictions for the following 5 houses:")
print(X.head())

print("The predicitons are")
print(melbourne_model.predict(X))

# Calculating the Mean Absolute Error [Error = Abs(Actual price - Prediction)]
predicted_home_prices = melbourne_model.predict(X)
errorMean = mean_absolute_error(y,predicted_home_prices)
print(errorMean)

# Split data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model_Ver2 = DecisionTreeRegressor()

# Fit model
melbourne_model_Ver2.fit(train_X, train_y)

# Predicted prices on validation data
val_predictions = melbourne_model_Ver2.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# Train model, return MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Recognising overfitting and underfitting models
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))