# Import modules
import sklearn 
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
        #kf = KFold(n_splits=n_splits, random_state=200, shuffle=False)
        kf = LeaveOneOut()

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
    

#Read input file
df = gpd.read_file("C:/Thesis/Analyse/analysefile_klein.shp")

#Start of ML model
    
#Define ML classifiers
names = ["Nearest Neighbors", "Gaussian Process", "Decision Tree"]
classifiers = [
    KNeighborsRegressor(n_neighbors=3,metric='minkowski')
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    #DecisionTreeClassifier(max_depth=5)
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


#Naive model and Classifier


#Iterate over classifiers
for name, clf in zip(names, classifiers):
    print('regressor quality: '+name+':'+str(myCVAScore(clf,X,y,cvn,dm=False)))




    


