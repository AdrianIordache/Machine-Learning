
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import xlsxwriter

import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


# In[4]:


def batch_normalization(features):
    for column in features:
        X = features[column].values[:, None]
        mini_batch_mean = X.sum(axis = 0) / len(X)
        mini_batch_var  = ((X - mini_batch_mean) ** 2).sum(axis = 0) / len(X)
        X_hat = (X - mini_batch_mean) / ((mini_batch_var + 1e-8) ** 0.5)
        features[column] = X_hat
        
    return features

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


# In[5]:


data = pd.read_csv("WineDescription.csv")
display(data.head(n=10))


# In[6]:


raw_features = data.drop('quality', axis = 1)
labels = data['quality']


# In[7]:


print ("Correlations between features in dataset:")
df = pd.DataFrame(raw_features, columns = raw_features.columns.tolist())
print (df.corr(method = "kendall"))
print('\n')


# In[8]:


min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(raw_features)


# In[9]:


#features = preprocessing.scale(raw_features)
#features = batch_normalization(raw_features)

# In[10]:


pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(features)
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


# In[11]:


def random_forest_param_selection(X, y):
    
    rfc = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [2625, 2650, 2675],
        'max_depth' :  [28,29,30, 27, 26],
    }
    
    folds = KFold(n_splits = 3, random_state = 42)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    clf = GridSearchCV(rfc, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    
    return clf.best_params_


# In[14]:


def ada_param_selection(X, y, clf):
    
    abc = AdaBoostClassifier(base_estimator = clf)
    
    param_grid = {
        'n_estimators': [20, 15, 17],
        'learning_rate' :  [0.5, 0.1, 0.7],
    }
    
    folds = KFold(n_splits = 3, random_state = 42)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    clf = GridSearchCV(abc, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    
    return clf.best_params_


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size = 0.75, test_size = 0.25, random_state = 42)

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

'''
dictionary = random_forest_param_selection(X_train, y_train)
n_estimators = dictionary['n_estimators']
max_depth = dictionary['max_depth']
clf1 = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
'''

clf1 = RandomForestClassifier(n_estimators = 2650, max_depth = 28, random_state=42)
abc = AdaBoostClassifier(n_estimators = 20, base_estimator = clf1, learning_rate=0.5)
abc.fit(X_train,y_train)
y_predict = abc.predict(X_test)
error = mean_squared_error(y_test, y_predict)
score = accuracy_score(y_test, y_predict)
print ("RandomForestClassifier Error:",error)
print ("RandomForestClassifier Score:",score)

test = pd.read_csv("Prediction.csv")
test.drop(['quality'], axis = 1, inplace = True)

new_test = min_max_scaler.fit_transform(test)

ada_predictions = abc.predict(new_test)

new_submission_file(predictions = ada_predictions, file_name = 'ada_boost_min_max_')

'''
Training set has 3298 samples.
Testing set has 1100 samples.
Training Score:  -0.4933374693247304
Best score: -0.49302607640994545
Best parameters: {'max_depth': 28, 'n_estimators': 2650}
RandomForestClassifier Error: 0.4309090909090909
RandomForestClassifier Score: 0.68

Training set has 3298 samples.
Testing set has 1100 samples.
Training Score:  -0.49424600876830177
Best score: -0.49181322013341416
Best parameters: {'learning_rate': 0.05, 'n_estimators': 20}
AdaBoostClassifier Error: 0.43363636363636365
AdaBoostClassifier Score: 0.6772727272727272
'''

