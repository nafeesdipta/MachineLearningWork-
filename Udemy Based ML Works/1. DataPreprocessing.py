# Data Preprocessing

# Importing the libraries
import tkinter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (Imputer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)

# Importing the dataset
dataset = pd.read_csv("/home/nafees/PycharmProjects/Test/Udemy Based ML Works/DataSet/DataPreprocessing.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Spliting dataset into training and test
X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
