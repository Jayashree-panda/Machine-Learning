#importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#fitting the regressor model to dataset


#predicting a new result \
y_pred = regressor.predict(6.5)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,Y)

#Visualizing the regression result(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot( X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or bluff(regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
