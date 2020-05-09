



#importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,0:3].values
Y = dataset.iloc[:,3].values



#Taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#fit matrix X to imputer object
imputer = imputer.fit(X[:,1:3])
#replace the missing data with mean
X[:,1:3] = imputer.transform(X[:,1:3])"""



#encoding categorical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
#fit it to matrix X to encodeit
X[:,0] = labelencoder_x.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
#fit it to matrix X to encodeit
Y = labelencoder_y.fit_transform(Y)"""


#splitting the data into test set and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)"""
