#prepara data
import pandas as pd

data_scaled= pd.read_csv('Flight_Dataset.csv')

# Set 'price' as the target variable
y = data_scaled['price']

# Extract the input features
X_data = data_scaled.drop(['price'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)

#Modelo

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression().fit(X_train, y_train)

y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)

accuracy_train = reg.score(X_train, y_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

accuracy_lr = reg.score(X_test, y_test)
mse_lr = mean_squared_error(y_test, y_pred_test)
r2_lr = r2_score(y_test, y_pred_test)

print("Accuracy - Train: {:} Test: {:}".format(accuracy_train, accuracy_lr))
print("MSE - Train: {:} Test: {:}".format(mse_train, mse_lr))
print("R2 - Train: {:} Test: {:}".format(r2_train, r2_lr))