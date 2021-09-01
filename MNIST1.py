#############################################################################################################
##
##  Author : Pakshal Shahikant Jain 
##  Date : 11/06/2021
##  Program : Machine Learning of MNIST Data set using Voting Classifier Alogortithm
##
############################################################################################################

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

def Voting(data_train,data_test,target_train,target_test) :
    log_clf = LogisticRegression(max_iter = 2000,random_state = 4)
    rnd_clf = RandomForestClassifier(n_estimators = 250,min_samples_split = 3,random_state = 4)
    knn_clf = KNeighborsClassifier(n_neighbors = 3,algorithm = 'ball_tree')

    vot_clf = VotingClassifier(estimators = [('lr',log_clf),('rnd',rnd_clf),('knn',knn_clf)],voting = 'soft')

    vot_clf.fit(data_train,target_train)

    print("Accuracy of Training Dataset is : ",vot_clf.score(data_train,target_train) * 100)
    print("Accuracy of Testing is : ",vot_clf.score(data_test,target_test) * 100)

def main() :
    Data = pd.read_csv("mnist.csv")
    print("Jay Ganesh..........")

    print("------------Information of Dataset---------------")
    print(Data.info())

    # Splitting of Data set into label and Feautures
    data = Data.iloc[:,1:] 
    target = Data.iloc[:,0]

    print("---Features---")
    print(data.head())
    print("---Labels---")
    print(target.head())

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.15,random_state = 42)

    Voting(data_train,data_test,target_train,target_test)
if __name__ == "__main__" :
    main()