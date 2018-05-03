# Import modules
import sklearn 
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from shapely.geometry import LineString






def turnintodistancematrix(X):
    #Calculate distance matrix
    return euclidean_distances(X,X) #Euclidean distance between all rows of X     
    #dist4 = pairwise_distances(X)   #Pairwise distances
 
    
    
def myCVAScore(classifier,X,y, n_splits, dm = False):
    #CMy validation function
        

        r2list=[]
        

        # remember to set n_splits and shuffle!
        kf = KFold(n_splits=n_splits, random_state=200, shuffle=False)
        #kf = LeaveOneOut()
        
        for train_index, test_index in kf.split(X, y):
            # assuming classifier object exists
            
            
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            
            if dm:
                X_train = turnintodistancematrix(X_train)

            X_test = X.iloc[test_index]
            if dm:
                X_test = turnintodistancematrix(X_test)
            y_test = y.iloc[test_index]

            # learn the classifier
            classifier.fit(X_train, y_train)

            # predict labels for test data
            predictions = classifier.predict(X_test)         
            
            r2 = sklearn.metrics.r2_score(predictions, y_test)
            print(r2)
            r2list.append(r2)
            
           # (precision, recall, f1, support) = precision_recall_fscore_support(y_test,predictions, average='weighted')
           # accuracylist.append(accuracy_score(y_test,predictions))
           # precisionlist.append(precision)
           # recalllist.append(recall)
           # f1list.append(f1)
            


        return np.asarray(r2list).mean()#(np.asarray(accuracylist).mean(),np.asarray(precisionlist).mean(), np.asarray(recalllist).mean(),np.asarray(f1list).mean(), np.asarray(coverageerrorlist).mean(), np.asarray(hamminglosslist).mean(), np.asarray(jaccardlist).mean())    


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt







    

#Read input file
df = gpd.read_file("C:/Thesis/Analyse/analysefile_klein.shp")

#Start of ML model
    
#Define ML classifiers
names = ["K Nearest Neighbors", "Gaussian Process", "Decision Tree", "Kernel Ridge", "Partial Least Squares", "Linear Regressor", "Support Vector"]
classifiers = [
    KNeighborsRegressor(n_neighbors=3,metric='minkowski'),
    GaussianProcessRegressor(kernel=None),
    DecisionTreeRegressor(max_depth=5),
    KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None),
    PLSRegression(n_components=2),
    LinearRegression(fit_intercept=True),
    SVR(kernel='rbf')
    
    ]

#Define Vector y and Matrix X
y = df[['r1_intens']]
X = df[['schoon_num', 'wegdek_num', 'INTENSITEI']]
#X = pd.DataFrame()
#X['X'] = df['geometry'].apply(lambda x: LineString(x).centroid.coords[:][0][0])  
#X['Y'] = df['geometry'].apply(lambda x: LineString(x).centroid.coords[:][0][1]) 


#Train/test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
  
#Define number of cross validations and classifier
cvn =5

#loo = LeaveOneOut
#loo.get_n_splits(X)

#for train_index, test_index in loo.split(X):
#   print("TRAIN:", train_index, "TEST:", test_index)
#   X_train, X_test = X[train_index], X[test_index]
#   y_train, y_test = y[train_index], y[test_index]

classes = list(set(y))


#Validation curves
title = "Learning Curves (KNN)"
cv = KFold(n_splits= 3, random_state=200, shuffle=False)
estimator =  KNeighborsRegressor(n_neighbors=3,metric='minkowski')
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

plt.show()


#Iterate over classifiers
for name, clf in zip(names, classifiers):
    print('regressor quality: '+name+':'+str(myCVAScore(clf,X,y,cvn,dm=False)))
    #print('MSE:' +str(mean_squared_error(y, y_test)))



    


