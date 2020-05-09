



#importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values



#feature scaling
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
Sc_Y = StandardScaler()
X = Sc_X.fit_transform(X)
Y = Sc_Y.fit_transform(Y)


#fitting SVR to dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#predicting a new result \
y_pred = Sc_Y.inverse_transform(regressor.predict(Sc_X.transform(np.array([[6.5]]))))

#Visualizing the SVR result

plt.scatter(X, Y, color = 'red')
plt.plot( X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff(regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
