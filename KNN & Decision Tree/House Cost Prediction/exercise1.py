# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from  sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.cross_validation import ShuffleSplit
from sklearn import neighbors



data = pd.read_csv('reg.csv')
print("Dataset contains ",data.shape[0]," samples with ",data.shape[1]," features")
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print("HERE!!!!")
print(features.columns.tolist())
new_features = features.drop(['PTRATIO'], axis = 1)


#The features, 'RM', 'LSTAT', and 'PTRATIO', give us quantitative information about
# each data point. The target variable, 'MEDV', will be the variable we seek to predict. These are stored in features and prices
#'RM' is the average number of rooms among homes in the neighborhood.
#'LSTAT' is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
#'PTRATIO' is the ratio of students to teachers in primary and secondary schools in the neighborhood.

# Calculate statistics
minimum_price = np.amin(prices)
maximum_price = np.amax(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

# Show the calculated statistics
print ("Statistics for dataset:\n")
print ("Minimum price: ${:,.2f}".format(minimum_price))
print ("Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))

X = np.array(features)
X = preprocessing.scale(X) 
y = np.array(prices)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, random_state=0)

# Decision Tree
regressor = DecisionTreeRegressor(max_depth=4)
regressor.fit(X_train,y_train)
y_predict=regressor.predict(X_test)
score = r2_score(y_test, y_predict)
print ("Score:",score)

# KNN
regressor2 = neighbors.KNeighborsRegressor(n_neighbors=4)
regressor2.fit(X_train,y_train)
y_predict2=regressor2.predict(X_test)
score = r2_score(y_test, y_predict2)
print ("Score:",score)


#Best Score DecisionTree 0.830517005946412 and KNN Score 0.8310125962676368 with Standard feautures

