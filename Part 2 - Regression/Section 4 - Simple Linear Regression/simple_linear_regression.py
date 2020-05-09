



#importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values





#splitting the data into test set and training set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
x_train = x_train.reshape((20,1))
#Fitting linear regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(x_train, y_train)
x_test = x_test.reshape((10,1))
#Predicting the test results
y_pred = regressor.predict(x_test)

#Visualizing the Training set result
"""plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()"""

#Visualizing the Test set result
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
