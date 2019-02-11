# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:54:29 2018

@author: Adrian Iordache
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn import svm
from matplotlib import style
from sklearn.preprocessing import StandardScaler
style.use("ggplot")


def svc_param_selection(X, y):
    Cs = [0.01, 0.001, 0.1, 1]
    gammas = [0.01, 0.1, 0.001, 1]
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    
    clf = GridSearchCV(svm.SVC(), param_grid)
    
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = None)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    
    return clf.best_params_


data = pd.read_csv('reg.csv')
print("Dataset contains ",data.shape[0]," samples with ",data.shape[1]," features")
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print ("Correlations between features in dataset:")
df = pd.DataFrame(features, columns = features.columns.tolist())
print (df.corr(method = "kendall"))
print('\n')




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


features['VAR_MAX'] = (maximum_price - data['MEDV']) 


print(features.columns.tolist());


X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2)

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

dictionary = svc_param_selection(X_train, y_train)
print(dictionary['C']) 
print(dictionary['gamma']) 
print(dictionary['kernel']) 
clf = svm.SVC(C = dictionary['C'], gamma = dictionary['gamma'], kernel = dictionary['kernel'])

#0.01
#0.01
#poly
#Test Score:  0.36127450980392156 without "VAR_MAX"
#clf = svm.SVC(C = 0.01, gamma = 0.01, kernel = 'poly')

clf.fit(X_train, y_train)
#score = clf.score(X_test, y_test)
score = (cross_val_score(clf, X_test, y_test, cv = None)).mean() 
print("Test Score: ", score)





