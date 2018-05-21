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





#Calculate distance matrix
def turnintodistancematrix(X):
    return euclidean_distances(X,X) #Euclidean distance between all rows of X     
    

#Define machine learning function    
def myCVAScore(classifier,X,y, n_splits, dm = False):
    #CMy validation function
        

        r2list=[]
        mselist=[]

        # remember to set n_splits and shuffle!
        kf = KFold(n_splits=n_splits, random_state=200, shuffle=False)
        #kf = LeaveOneOut(n_splits=n_splits, shuffle=False)
        
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
            mse = mean_squared_error(predictions, y_test)
            print('r2 score: ' +str(r2))
            print('mse: ' +str(mse))
            r2list.append(r2)
            mselist.append(mse)

            
        return np.asarray(r2list).mean(), np.asarray(mselist).mean()#(np.asarray(accuracylist).mean(),np.asarray(precisionlist).mean(), np.asarray(recalllist).mean(),np.asarray(f1list).mean(), np.asarray(coverageerrorlist).mean(), np.asarray(hamminglosslist).mean(), np.asarray(jaccardlist).mean())    

#Read input file
df = gpd.read_file("C:/Thesis_analysis/analysis_file.shp")


#Start of ML model
    
#Define ML classifiers
names = ["K Nearest Neighbors", "Gaussian Process", "Decision Tree", "Kernel Ridge", "Partial Least Squares", "Linear Regressor", "Support Vector"]
classifiers = [
    KNeighborsRegressor(n_neighbors=3,metric='precomputed'),
    GaussianProcessRegressor(kernel=None),
    DecisionTreeRegressor(max_depth=5),
    KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None),
    PLSRegression(n_components=2),
    LinearRegression(fit_intercept=True),
    SVR(kernel='rbf')
    
    ]

#Define Vector y and Matrix X
y = df[['intens_cor']]
X = df[['schoon_num', 'wegdek_num', 'brid_int']]

X['X'] = df['geometry'].apply(lambda x: LineString(x).centroid.coords[:][0][0])  
X['Y'] = df['geometry'].apply(lambda x: LineString(x).centroid.coords[:][0][1]) 

  
#Define number of cross validations and classifier
cvn = 5


classes = list(set(y))


#Iterate over classifiers
for name, clf in zip(names, classifiers):
    print('R2 regressor quality(mean), MSE(mean) '+name+':'+str(myCVAScore(clf,X,y,cvn,dm=False)))

