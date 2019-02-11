# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


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


data = pd.read_csv("clf.csv")
display(data.head(n=10))


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

print(features_raw.columns.tolist());

relevant_features_decision_tree = features_raw.drop(['age', 'sex', 'race', 'workclass', 'hours-per-week'], axis = 1)
relevant_features_knn = features_raw.drop(['age', 'sex', 'race', 'workclass', 'occupation', 'hours-per-week', 'education_level', 'native-country'], axis = 1)

#print(relevant_features_decision_tree.columns.tolist());
#print(relevant_features_knn.columns.tolist());


print("HERE!!!!")
#print(relevant_features_decision_tree.head())
#print(relevant_features_knn.head())

scaler = MinMaxScaler()
numerical_decision_tree = ['capital-gain', 'capital-loss']
numerical_knn = ['capital-gain', 'capital-loss']

features_log_minmax_transform = pd.DataFrame(data = relevant_features_knn)
features_log_minmax_transform[numerical_knn] = scaler.fit_transform(data[numerical_knn])


#Convert categorical variable into dummy/indicator variables
features = pd.get_dummies(relevant_features_knn)
income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)

'''X = np.array(features)
X = preprocessing.scale(X) 
y = np.array(income)'''

X = features
y = income

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

clf1 = DecisionTreeClassifier(random_state=0)
clf1.fit(X_train,y_train)
y_predict=clf1.predict(X_test)
score = accuracy_score(y_test, y_predict)
print ("DecisionTreeClassifier Score:",score)

clf2 = KNeighborsClassifier()
clf2.fit(X_train,y_train)
y_predict=clf2.predict(X_test)
score = accuracy_score(y_test, y_predict)
print ("KNeighborsClassifier Score:",score)

# Best Score DecisionTree 0.8653421633554084 and KNN Score 0.8344370860927153, test_size = 0.01 with scaling
# Best Score KNN 0.8543046357615894 and DecisionTree Score 0.847682119205298, test_size = 0.01 without scaling