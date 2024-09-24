#====================libraries======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#====================libraries======================

#loading from csv file(change the file name if you want to)
df = pd.read_csv('Breast-cancer.csv')

#make a list of data's we want.
X = df[["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1",
        "Classification"]].values
y = df['Classification'].values

#normalizing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#Train Model and make a test set.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#definding knn for cheking tests.
for k in range(5,8,2):
        neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
        yhat = neigh.predict(X_test)
        #seeing the results!
        print(f"if k={k}:")
        print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
        print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
        print("Train set fscore: ", metrics.f1_score(y_train, neigh.predict(X_train)))
        print("Test set fscore: ", metrics.f1_score(y_test, yhat),
              "\n\n===========================\n")