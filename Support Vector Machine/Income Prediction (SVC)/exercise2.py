# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:33:07 2018

@author: Adrian Iordache
"""


import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle


# The modified census dataset consists of approximately 32,000 data points, 
#with each datapoint having 13 features

#Features

#age: Age
#workclass: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
#education_level: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
#education-num: Number of educational years completed
#marital-status: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
#occupation: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
#relationship: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
#race: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
#sex: Sex (Female, Male)
#capital-gain: Monetary Capital Gains
#capital-loss: Monetary Capital Losses
#hours-per-week: Average Hours Per Week Worked
#native-country: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)


#Target Variable

#income: Income Class (<=50K, >50K)

def svc_param_selection(X, y):
    Cs = [0.1, 1, 10, 100, 1000]
    gammas = [0.01, 0.1, 0.001]
    kernels = ['rbf', 'linear', 'poly']
    
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    
    clf = GridSearchCV(svm.SVC(), param_grid, cv = 5)
    
    cross_score = cross_val_score(estimator = clf, X = X, y = y)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    
    return clf.best_params_


data = pd.read_csv("clf.csv")
display(data.head(n=10))

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

features_raw.drop([ 'relationship', 'occupation', 'education_level', 'native-country'], axis = 1, inplace = True)

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_raw)
features_log_minmax_transform[numerical] = scaler.fit_transform(data[numerical])


#Convert categorical variable into dummy/indicator variables
features = pd.get_dummies(features_raw)
income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)


print ("Correlations between features in dataset:")
df = pd.DataFrame(features, columns = features.columns.tolist())
print (df.corr(method = "kendall"))
print('\n')

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
cax = ax1.imshow(df.corr(method = "kendall"), interpolation="nearest", cmap=cmap)
ax1.grid(True)
plt.title('Correlations between features in dataset:')
labels = features.columns.tolist()
ax1.set_xticklabels(labels, fontsize=5)
ax1.set_yticklabels(labels, fontsize=5)
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()

#print(features.columns.tolist());

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, train_size = 0.8, test_size = 0.2, random_state = 0)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#Debugging purposes... finding the best parameters for train_size = 0.08, test_size = 0.02, and then we will use the for all data
'''dictionary = svc_param_selection(X_train, y_train)
print(dictionary['C']) #1000
print(dictionary['gamma']) #0.01
print(dictionary['kernel']) #rbf 
clf = svm.SVC(C = dictionary['C'], gamma = dictionary['gamma'], kernel = dictionary['kernel'])'''

'''clf = svm.SVC(C = 1000, gamma = 0.01, kernel = 'rbf' )
clf.fit(X_train, y_train)

with open('svc.pickle', 'wb') as f:
    pickle.dump(clf, f)'''
    
pickle_in = open('svc.pickle', 'rb')
clf = pickle.load(pickle_in)

score = clf.score(X_test, y_test) 
print("Test Score: ", score)
#Test Score: 0.8468767274737424
