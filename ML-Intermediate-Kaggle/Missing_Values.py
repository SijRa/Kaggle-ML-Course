import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

melb_data = pd.read_csv('melb_data.csv')

y = melb_data.Price

# Only using numerical predictions 
melb_predictors = melb_data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

def score_dataset(train_X, val_X, train_y, val_y):
    """
    Return mean absolute error
    """
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(val_y, predictions)

# Get columns with missing values
columns_with_missing_values = [col for col in train_X.columns if train_X[col].isnull().any()]

reduced_train_X = train_X.drop(columns_with_missing_values, axis=1)
reduced_val_X = val_X.drop(columns_with_missing_values, axis=1)

print('--MAE without imputation--')
print(score_dataset(reduced_train_X, reduced_val_X, train_y, val_y))

# To impute missing values
from sklearn.impute import SimpleImputer

# Imputation
imputor = SimpleImputer()
imputed_train_X = pd.DataFrame(imputor.fit_transform(train_X))
imputed_val_X = pd.DataFrame(imputor.transform(val_X))

# Fix columns removed by imputation 
imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns

print('\n--MAE with imputation--')
print(score_dataset(imputed_train_X,imputed_val_X, train_y, val_y))

train_X_copy = train_X.copy()
val_X_copy = val_X.copy()

# New columns indicating imputed rows
for col in columns_with_missing_values:
    train_X_copy[col + '_was_missing'] = train_X_copy[col].isnull()
    val_X_copy[col + '_was_missing'] = val_X_copy[col].isnull()

# Imputation approach 2
imputer2 = SimpleImputer()
imputed_train_X_copy = pd.DataFrame(imputer2.fit_transform(train_X_copy))
imputed_val_X_copy = pd.DataFrame(imputer2.transform(val_X_copy))

imputed_train_X_copy.columns = train_X_copy.columns
imputed_val_X_copy.columns = val_X_copy.columns

print('\n--MAE with imputation and missing column--')
print(score_dataset(imputed_train_X_copy, imputed_val_X_copy, train_y, val_y))

# Rows and Columns
print(train_X.shape)

# Number of missing values
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])