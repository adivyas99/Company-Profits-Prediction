## Predicting Salaries of a Company by using 
## Multiple Linear Regression
## Using sample Dataset

# 1. Importing Libraries --

import numpy as np
#For Numerical calculations

import matplotlib.pyplot as plt
# For Data Vizualization

import pandas as pd
# For Data Management

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# For Labelling and Encoding Data

from sklearn.cross_validation import train_test_split
# For Spliting the Dataset

from sklearn.linear_model import LinearRegression
# For Performing Linear Regression

# 2. Importing Dataset --

dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:, :-1].values # Features set
y = dataset.iloc[:, 4].values # Labels set

# 3. Encoding all categorical Attributes --

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  # For avoiding Dummy Variable

# 5. Splitting dataset into Training and Test set --

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# With 20% Test Set
# Feature Scaling is NOT performed here

# 6. Fitting Multiple Linear Regression to Training set --

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 7. Predicting the Test set Results --

y_pred = regressor.predict(X_test)
