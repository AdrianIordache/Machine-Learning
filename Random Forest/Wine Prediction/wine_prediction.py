# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:13:58 2018

@author: Adrian Iordache

"""
import numpy as np
import pandas as pd
import xlsxwriter


import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def batch_normalization(features):
    for column in features:
        X = features[column].values[:, None]
        mini_batch_mean = X.sum(axis = 0) / len(X)
        mini_batch_var  = ((X - mini_batch_mean) ** 2).sum(axis = 0) / len(X)
        X_hat = (X - mini_batch_mean) / ((mini_batch_var + 1e-8) ** 0.5)
        features[column] = X_hat
        
    return features

def decision_tree_param_selection(X, y):
    
    algorithm = DecisionTreeClassifier()
    param_grid = {
         'max_depth' : list(range(1, 30)),
         'criterion': ['entropy', 'gini'],
         'min_samples_leaf' :[1, 2, 3, 4, 5],
         'min_samples_split':list(range(2, 10)),
     }

    
    folds = 3
    scorer = make_scorer(accuracy_score)
    clf = GridSearchCV(algorithm, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    
    return clf.best_params_

def random_forest_param_selection(X, y):
    
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [3000 ,3500, 4000],
        'max_depth' :  [35, 45, 50],
    }
    
    folds = 3
    scorer = make_scorer(accuracy_score)
    clf = GridSearchCV(rfc, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    
    return clf.best_params_

def knn_param_selection(X, y):
    k_range = list(range(1, 100))
    weight_options = ["uniform", "distance"]
    scorer = make_scorer(accuracy_score)
    
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    algorithm = KNeighborsClassifier()
    
    folds = 5
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

'''print ("Correlations between features in dataset:")
df = pd.DataFrame(features, columns = features.columns.tolist())
print (df.corr(method = "kendall"))
print('\n')'''

'''pca = PCA(n_components = 2)
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
ax.grid()'''

X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size = 0.75, test_size = 0.25, random_state = 42)

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

clf1 = RandomForestClassifier(n_estimators = 3500, max_depth = 40, random_state=42)
clf1.fit(X_train,y_train)
y_predict = clf1.predict(X_test)
error = mean_squared_error(y_test, y_predict)
score = accuracy_score(y_test, y_predict)
print ("RandomForestClassifier Error:",error)
print ("RandomForestClassifier Score:",score)

test = pd.read_csv("Prediction.csv")
test.drop(['quality'], axis = 1, inplace = True)

NewX = batch_normalization(test)

predictions = clf1.predict(NewX)

print(predictions)

workbook = xlsxwriter.Workbook('submission_file.xlsx')
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



#Finding best parameters for various classifiers with GridSearch

'''dictionary = random_forest_param_selection(X_train, y_train)
n_estimators = dictionary['n_estimators']
max_depth = dictionary['max_depth']
clf1 = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
clf1.fit(X_train,y_train)
y_predict = clf1.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("RandomForestClassifier Error:", error)
print ("RandomForestClassifier Score:", score)'''

#Training Score:  0.6252367366771775
#Best score: 0.626137052759248
#Best parameters: {'max_depth': 40, 'n_estimators': 3500}
#RandomForestClassifier Error: 0.47
#RandomForestClassifier Score: 0.6654545454545454

'''dictionary = knn_param_selection(X_train, y_train)
k_neighbors = dictionary['n_neighbors']
k_weights = dictionary['weights']
clf2 = KNeighborsClassifier(n_neighbors = k_neighbors, weights = k_weights)
clf2.fit(X_train,y_train)
y_predict=clf2.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("KNeighborsClassifier Error:", error)
print ("KNeighborsClassifier Score:",score)'''

#Training Score:  0.5197296765043609
#Best score: 0.6400848999393571
#Best parameters: {'n_neighbors': 62, 'weights': 'distance'}
#KNeighborsClassifier Error: 0.44181818181818183
#KNeighborsClassifier Score: 0.6563636363636364

'''dictionary = decision_tree_param_selection(X_train, y_train)
max_depth = dictionary['max_depth']
criterion = dictionary['criterion']
min_samples_leaf = dictionary['min_samples_leaf']
min_samples_split = dictionary['min_samples_split']
clf3 = DecisionTreeClassifier(max_depth = max_depth, criterion = criterion, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)
clf3.fit(X_train,y_train)
y_predict=clf3.predict(X_test)
score = accuracy_score(y_test, y_predict)
error = mean_squared_error(y_test, y_predict)
print ("DecisionTreeClassifier Error:", error)
print ("DecisionTreeClassifier Score:",score)'''

#Training Score:  0.5164086572244317
#Best score: 0.5433596118859915
#Best parameters: {'criterion': 'gini', 'max_depth': 14, 'min_samples_leaf': 1, 'min_samples_split': 2}
#DecisionTreeClassifier Error: 0.7281818181818182
#DecisionTreeClassifier Score: 0.5790909090909091

