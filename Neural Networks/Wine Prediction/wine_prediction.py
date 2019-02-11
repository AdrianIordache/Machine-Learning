# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:13:58 2018

@author: Adrian Iordache

"""
import numpy as np
import pandas as pd
import xlsxwriter
import pickle


import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


def new_submission_file(predictions, file_name):
    workbook = xlsxwriter.Workbook(file_name + 'submission.xlsx')
    worksheet = workbook.add_worksheet()
    
    row = 0
    col = 0
    
    worksheet.write(row, col, 'Index')
    worksheet.write(row, col + 1, 'Quality')
    
    row += 1
    
    prediction_list = predictions.tolist()
    
    
    for i in range(predictions.size):
        worksheet.write(row, col, i + 1)
        worksheet.write(row, col + 1, prediction_list[i])
        row += 1
    
    workbook.close()
    
    

def batch_normalization(features):
    for column in features:
        X = features[column].values[:, None]
        mini_batch_mean = X.sum(axis = 0) / len(X)
        mini_batch_var  = ((X - mini_batch_mean) ** 2).sum(axis = 0) / len(X)
        X_hat = (X - mini_batch_mean) / ((mini_batch_var + 1e-8) ** 0.5)
        features[column] = X_hat
        
    return features

def mlp_param_selection(X, y):
    
    algorithm = MLPClassifier(random_state = 42)
    param_grid = {
          'activation' : ['relu', 'identity', 'logistic', 'tanh'],    
          'solver': ['lbfgs', 'sgd', 'adam'], 
          'learning_rate' : ['adaptive', 'constant'],
          'alpha': 10.0 ** - np.arange(2, 4), 
          'hidden_layer_sizes': [700, 1000, 1200], 
          'max_iter': [1000], 
     }

    
    folds = KFold(n_splits=3, random_state=42)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    clf = GridSearchCV(algorithm, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    
    return clf.best_params_


def knn_param_selection(X, y):
    k_range = list(range(1, 200))
    weight_options = ["uniform", "distance"]
    scorer = make_scorer(accuracy_score)
    
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    algorithm = KNeighborsClassifier()
    
    folds = KFold(n_splits = 5, random_state=42)
    scorer = make_scorer(mean_squared_error, greater_is_better = False)
    grid_search = GridSearchCV(algorithm, param_grid, cv = folds, scoring = scorer)
    grid_search.fit(X, y.ravel())
    
    cross_score = cross_val_score(estimator = algorithm, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    return grid_search.best_params_


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

data = pd.read_csv("WineDescription.csv")
#display(data.head(n=10))

features = data.drop('quality', axis = 1)

#features.drop(['sulphates', 'volatile acidity'], axis = 1, inplace = True)

labels = data['quality']

X = batch_normalization(features)

#min_max_scaler = MinMaxScaler()
#X = min_max_scaler.fit_transform(features)
#X = preprocessing.scale(features)

print ("Correlations between features in dataset:")
df = pd.DataFrame(features, columns = features.columns.tolist())
print (df.corr(method = "kendall"))
print('\n')

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['Principal component 1', 'Principal component 2'])

finalDf = pd.concat([principalDf, data[['quality']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = np.unique(labels.values)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal component 1']
               , finalDf.loc[indicesToKeep, 'Principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size = 0.75, test_size = 0.25, random_state = 42)

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

clf1 = MLPClassifier(activation = 'relu', alpha = 0.00001, hidden_layer_sizes = 900, learning_rate = 'adaptive', max_iter = 1000, solver = 'adam', random_state = 42)
clf1.fit(X_train,y_train)
y_predict = clf1.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("MLPClassifier Error:", error)
print ("MLPClassifier Score:",score)

clf2 = KNeighborsClassifier(n_neighbors = 62, weights = 'distance')
clf2.fit(X_train,y_train)
y_predict = clf2.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("KNeighborsClassifier Error:", error)
print ("KNeighborsClassifier Score:",score)

test = pd.read_csv("Prediction.csv")
test.drop(['quality'], axis = 1, inplace = True)

NewX = batch_normalization(test)

neural_net_predictions = clf1.predict(NewX)
knn_predictions = clf2.predict(NewX)

new_submission_file(predictions = neural_net_predictions, file_name = 'neural_net_')
new_submission_file(predictions = knn_predictions, file_name = 'best_knn_')





#Finding best parameters for various classifiers with GridSearch

''''
dictionary = mlp_param_selection(X_train, y_train)
activation = dictionary['activation']
solver = dictionary['solver']
learning_rate = dictionary['learning_rate']
alpha = dictionary['alpha']
hidden_layer_sizes = dictionary['hidden_layer_sizes']
max_iter = dictionary['max_iter']

clf1 = MLPClassifier(activation = activation, solver = solver, learning_rate = learning_rate, max_iter = max_iter, alpha = alpha, hidden_layer_sizes = hidden_layer_sizes, random_state = 42)
clf1.fit(X_train,y_train)
y_predict = clf1.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("MLPClassifier Error:", error)
print ("MLPClassifier Score:",score)
'''

#Best score: 0.6000606428138265
#Best parameters: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 1200, 'learning_rate': 'adaptive', 'max_iter': 1000, 'solver': 'adam'}
#MLPClassifier Error: 0.6036363636363636
#MLPClassifier Score: 0.5718181818181818


'''
dictionary = knn_param_selection(X_train, y_train)
k_neighbors = dictionary['n_neighbors']
k_weights = dictionary['weights']

clf2 = KNeighborsClassifier(n_neighbors = k_neighbors, weights = k_weights)
clf2.fit(X_train,y_train)
y_predict = clf2.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("KNeighborsClassifier Error:", error)
print ("KNeighborsClassifier Score:",score)
'''

#Training Score:  0.5197296765043609
#Best score: 0.6400848999393571
#Best parameters: {'n_neighbors': 62, 'weights': 'distance'}
#KNeighborsClassifier Error: 0.4381818181818182
#KNeighborsClassifier Score: 0.6681818181818182


#Random Forest with AdaBoost

'''
clf1 = RandomForestClassifier(n_estimators = 3500, max_depth = 40, random_state=42)
abc = AdaBoostClassifier(n_estimators = 10, base_estimator = clf1, learning_rate=1)
abc.fit(X_train,y_train)
y_predict = abc.predict(X_test)
error = mean_squared_error(y_test, y_predict)
score = accuracy_score(y_test, y_predict)
print ("RandomForestClassifier Error:",error)
print ("RandomForestClassifier Score:",score)
'''
#RandomForestClassifier Error: 0.44
#RandomForestClassifier Score: 0.6763636363636364




