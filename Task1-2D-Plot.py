import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# read data from .csv in folder
df = pd.read_csv("houseprice_data.csv")

X = df.iloc[:, [9]].values # input grade
y = df.iloc[:, 0].values # target variable price

# visualise initial data set
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, color = 'blue', marker = '*')
ax1.set_title('Initial Data set (Price VS Grade)')
ax1.set_xlabel('Grade')
ax1.set_ylabel('Price')

fig1.tight_layout()

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# visualise training set results
fig2, ax2 = plt.subplots()

ax2.scatter(X_train, y_train)
# ax2.plot(X_train, regr.predict(X_train), color = 'red', marker = '*')
ax2.set_title('Train set (Price VS Grade)')
ax2.set_xlabel('Grade')
ax2.set_ylabel('Price')

fig2.tight_layout()

# visualise test set results
fig3, ax3 = plt.subplots()

ax3.scatter(X_test, y_test)
# ax3.plot(X_test, regr.predict(X_test))
ax3.set_title('Test Set (Price VS Grade)')
ax3.set_xlabel('Grade')
ax3.set_ylabel('Price')

fig3.tight_layout()

# The coefficients
print('Coefficients: ', regr.coef_)
# The intercept
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.4f' % mean_squared_error(y_test, regr.predict(X_test)))
# The R^2 value or the coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_test, regr.predict(X_test)))

print('Predict single value: ', regr.predict(np.array([[6]])))