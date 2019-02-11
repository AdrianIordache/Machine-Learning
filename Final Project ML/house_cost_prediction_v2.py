###############################################################
#  Machine Learning idrive Global                             #
#                                                             #
#  @package     Proiect Machine Learning                      #
#  @authors     Adrian Iordache                               #
#  @license     DO NOT redistribute                           #
#                                                             #
###############################################################

# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import decomposition
from sklearn import svm


# In[2]:


def RMSLE(ypred, ytest): 
    assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))


# In[3]:


def visualize_data(data):
    pca = decomposition.PCA()
    pca.n_components = 2
    pca_data = pca.fit_transform(data)

    #print("Shape of PCA = ", pca_data.shape)

    pca_data = np.vstack((pca_data.T, labels)).T

    #print("Shape of PCA after combine = ", pca_data.shape)

    pca_df = pd.DataFrame(data = pca_data, columns = ("First Principal", "Second Principal", "Label"))

    for idx in range(len(pca_df)):
        for counter in range(100):
            lower_price = counter * 10000
            upper_price = (counter + 1) * 10000
            if pca_df.at[idx, "Label"] > lower_price and pca_df.at[idx, "Label"] < upper_price:
                pca_df.at[idx, "Label"] = upper_price
                break

    sns.FacetGrid(pca_df, hue = "Label", size = 10).map(plt.scatter, "First Principal", "Second Principal").add_legend()
    plt.show()


# In[4]:


def solve_nan_values(data):
    limit = int(len(data) * 0.75)
    
    for columns in data.columns.tolist():
        if pd.isnull(data[columns]).any(axis = 0) == True:
            no_of_na = 0
            for value in data[columns]:
                if pd.isna(value) == True:
                    no_of_na += 1

            #print(columns)
            #print("{} missing values from {}.".format(no_of_na, int(len(data))))

            if no_of_na > limit:
                #print("Drop")
                data.drop(columns, axis = 1, inplace = True)
            else:
                #print("Replace")
                value = data[columns].value_counts().index.tolist()
                data = data.fillna({columns: value[0]})
    
    
    return data


# In[5]:


def convert_to_numerical(data):
    for columns in data.columns.tolist():
        categories = data[columns]
        data.drop(columns, axis = 1, inplace = True)
        dummy = pd.get_dummies(categories)
        data = pd.concat([data, dummy], axis = 1)
    
    return data
    


# In[6]:


def drop_highly_correlated_feautres(data):
    corr = data.corr()
    sns.heatmap(corr)

    # Create correlation matrix
    corr_matrix = raw_features.corr().abs()

    #print(corr_matrix)

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.90
    to_high = [column for column in upper.columns if any(upper[column] > 0.90)]

    # Drop features 
    data.drop(to_high, axis = 1, inplace = True)
    
    return data


# In[7]:


def knn_param_selection(X, y):
    print("START")
    k_range = list(range(1, 100))
    weight_options = ["uniform", "distance"]
    
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    algorithm = KNeighborsRegressor()
    
    folds = KFold(n_splits = 5, random_state=42)
    scorer = make_scorer(RMSLE, greater_is_better = False)
    clf = GridSearchCV(algorithm, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = algorithm, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y.ravel())
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    print("STOP")
    return clf.best_params_


# In[8]:


def svr_param_selection(X, y):
    print("START")
    Cs = [0.01, 0.001, 0.1, 1]
    gammas = [0.01, 0.1, 0.001, 1]
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    algorithm = svm.SVR()
    
    
    folds = KFold(n_splits = 5, random_state=42)
    scorer = make_scorer(RMSLE, greater_is_better = False)
    clf = GridSearchCV(algorithm, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = algorithm, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y.ravel())
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    print("STOP")
    return clf.best_params_


# In[9]:


def random_forest_param_selection(X, y):
    print("START")
    rfr = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [745, 750, 755],
        'max_depth' :  [15 ,20 ,25],
    }
    
    folds = KFold(n_splits = 5, random_state = 42)
    scorer = make_scorer(RMSLE, greater_is_better=False)
    clf = GridSearchCV(rfr, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    print("STOP")
    return clf.best_params_


# In[10]:


def mlp_param_selection(X, y):
    print("START")
    algorithm = MLPRegressor(random_state = 42)
    param_grid = {
          'activation' : ['relu', 'logistic', 'tanh'],    
          'solver': ['lbfgs', 'adam'], 
          'learning_rate' : ['adaptive'],
          'alpha': 10.0 ** - np.arange(2, 4), 
          'hidden_layer_sizes': [700, 1000, 1200], 
          'max_iter': [1000], 
     }

    
    folds = KFold(n_splits=3, random_state=42)
    scorer = make_scorer(RMSLE, greater_is_better=False)
    clf = GridSearchCV(algorithm, param_grid, cv = folds, scoring = scorer)
    cross_score = cross_val_score(estimator = clf, X = X, y = y, cv = folds)
    meanScore = np.average(cross_score)
    print("Training Score: ", meanScore)
    
    clf.fit(X, y)
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    print("STOP")
    return clf.best_params_


# In[11]:

#Reading Data From File
data = pd.read_csv("train.csv")
data = data.drop('Id', axis = 1)
#display(data.head(n=20))

#Creating features and labels
raw_features = data.drop("SalePrice", axis = 1)
labels = data["SalePrice"]
display(raw_features.head(n=10))


# In[12]:

#Solve NA Values from dataset
raw_features = solve_nan_values(data = raw_features)
raw_features.head(n = 10)


# In[13]:

#Select Categorical Data
categorical_data = raw_features.select_dtypes(include = ['object']).copy()
categorical_data.head(n = 10)

#Convert categorical data to numerical
categorical_data = convert_to_numerical(data = categorical_data)
categorical_data.head(n = 10)


# In[14]:

#Remove Categorical Data from dataset
raw_features.drop(raw_features.select_dtypes(['object']), inplace = True, axis = 1)

#Merge old categorical data with the rest of the features
raw_features = pd.concat([raw_features, categorical_data], axis = 1)

raw_features.head(n = 10)

#Visualize Data using PCA
visualize_data(data = raw_features)

#Drop Highly correlated features
raw_features = drop_highly_correlated_feautres(data = raw_features)


# In[15]:

#Preprocessing Step and Visualize again
x = raw_features.iloc[:,:-1].values
standard_scaler = preprocessing.StandardScaler()
features = standard_scaler.fit_transform(x)
visualize_data(data = features)


# In[16]:

#Training and testing with hyperparameter tunning using GridSearch and the results 
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size = 0.75, test_size = 0.25, random_state = 42)

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#dictionary = random_forest_param_selection(X_train, y_train)
#n_estimators = dictionary['n_estimators']
#max_depth = dictionary['max_depth']

#n_estimators = n_estimators, max_depth = max_depth
#n_estimators = 1000, max_depth = 50

regressor = RandomForestRegressor(n_estimators = 25, max_depth = 750)
ada = AdaBoostRegressor(n_estimators = 20, base_estimator = regressor, learning_rate = 0.5)
ada.fit(X_train,y_train)
y_predict = ada.predict(X_test)
error = RMSLE(y_test, y_predict)
print ("RandomForestRegressor Error:",error)
#RandomForestRegressor Error: 0.13995298236641165


#Training Score:  -0.15247668929852423
#Best score: -0.1523665243952441
#Best parameters: {'max_depth': 25, 'n_estimators': 750}
#RandomForestRegressor Error: 0.14002055610418848


# In[17]:


print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#dictionary = knn_param_selection(X_train, y_train)
#k_neighbors = dictionary['n_neighbors']
#k_weights = dictionary['weights']
#n_neighbors = k_neighbors, weights = k_weights

regressor = KNeighborsRegressor(n_neighbors = 13, weights = 'distance')
regressor.fit(X_train,y_train)
y_predict = regressor.predict(X_test)
error = RMSLE(y_test, y_predict)
print ("KNeighborsRegressor Error:", error)

#Training Score:  0.6751081074062033
#Best score: -0.2035467062998057
#Best parameters: {'n_neighbors': 13, 'weights': 'distance'}
#KNeighborsRegressor Error: 0.20294638416160402


# In[18]:


print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#dictionary = svr_param_selection(X_train, y_train) 
#C = dictionary['C']
#gamma = dictionary['gamma']
#kernel = dictionary['kernel']
#C = C, gamma = gamma, kernel = kernel

regressor = svm.SVR(C = 0.001, gamma = 1, kernel = 'poly')
regressor.fit(X_train,y_train)
y_predict = regressor.predict(X_test)
error = RMSLE(y_test, y_predict)
print ("Support Vector Machine Error:", error)

#Training Score:  -0.049315027243430176
#Best score: -0.3414677011904274
#Best parameters: {'C': 0.001, 'gamma': 1, 'kernel': 'poly'}
#Support Vector Machine Error: 0.3427940440637342


# In[19]:


print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#dictionary = mlp_param_selection(X_train, y_train)
#activation = dictionary['activation']
#solver = dictionary['solver']
#learning_rate = dictionary['learning_rate']
#alpha = dictionary['alpha']
#hidden_layer_sizes = dictionary['hidden_layer_sizes']
#max_iter = dictionary['max_iter']
#activation = activation, solver = solver, learning_rate = learning_rate, max_iter = max_iter, alpha = alpha, hidden_layer_sizes = hidden_layer_sizes, random_state = 42

regressor = MLPRegressor(activation = 'logistic', solver = 'lbfgs', learning_rate = 'adaptive', max_iter = 1000, alpha = 0.001, hidden_layer_sizes = 700, random_state = 42)
regressor.fit(X_train,y_train)
y_predict = regressor.predict(X_test)
error = RMSLE(y_test, y_predict)
print ("MLPRegressor Error:", error)

#Training Score:  -0.18151158315872673
#Best score: -0.1814913343616067
#Best parameters: {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': 700, 'learning_rate': 'adaptive', 'max_iter': 1000, 'solver': 'lbfgs'}
#MLPRegressor Error: 0.16807183194855363

