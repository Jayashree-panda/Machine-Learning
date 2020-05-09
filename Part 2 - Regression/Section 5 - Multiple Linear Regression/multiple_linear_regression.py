



#importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
#fit it to matrix X to encodeit
X[:,3] = labelencoder_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#handling dummy trap
X = X[:,1:]

#splitting the data into test set and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



#fitting multiple linear regression into the test set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#predicting test data
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward elimination(removing independent variables which are not statistically important)
import statsmodels.formula.api as sm
#adding column of 1s(x0)
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
#OLS(ordinary least squares)significance level is 0.01
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

#removing variable with highest p value
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())