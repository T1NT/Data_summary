
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# Load data
train_data_path = 'train.csv'
test_data_path = 'test.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Preprocessing: Handling missing values and encoding categorical variables
categorical_columns = train_df.select_dtypes(include=['object']).columns
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(train_df.median(), inplace=True)
train_df = pd.get_dummies(train_df, columns=categorical_columns)
test_df = pd.get_dummies(test_df, columns=categorical_columns)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# Feature scaling
scaler = MinMaxScaler()
features = train_df.columns.drop(['Id', 'SalePrice'])
train_df[features] = scaler.fit_transform(train_df[features])
test_df[features] = scaler.transform(test_df[features])

# Feature engineering
train_df['TotalSqFt'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
train_df['Age'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['TotalSqFt'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']
test_df['Age'] = test_df['YrSold'] - test_df['YearBuilt']

# Model training and tuning
X_train = train_df[features]
y_train = train_df['SalePrice']
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predicting and creating submission file
X_test = test_df[features]
predictions = model.predict(X_test)
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)
