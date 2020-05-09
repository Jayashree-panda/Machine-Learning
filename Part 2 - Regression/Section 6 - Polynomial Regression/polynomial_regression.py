



#importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values



#Fitting linear regression into training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualizing the linear regression result
plt.scatter(X, Y, color = 'red')
plt.plot( X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff(linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the polynomial regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot( X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff(linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting a new result using linear regression
print(lin_reg.predict(6.5))

#predicting a new result using polynomial regression
print(lin_reg2.predict(poly_reg.fit_transform(6.5)))