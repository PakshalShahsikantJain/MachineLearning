########################################################################################################################
##
##  Author : Pakshal Shashikant Jain
##  Date : 02/06/2021
##  Program : Python - Machine Learning : Ensemble machine learning with MNIST dataset 
##
#########################################################################################################################


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

def DecisionTree(data_train,data_test,target_train,target_test) :
    print("\n---------------------------Decision Tree Classifier Alogorithm------------------------------------")
    dobj = DecisionTreeClassifier()

    dobj.fit(data_train,target_train)

    print("Training Accuracy using Decision Tree Classifier is : ",dobj.score(data_train,target_train) * 100)
    print("Testing Accuracy using Decision Tree Classifier is : ",dobj.score(data_test,target_test) * 100)

def RandomForest(data_train,data_test,target_train,target_test) :
    print("\n---------------------------Random Forest Classifier Alogorithm------------------------------------")
    robj = RandomForestClassifier(n_estimators = 100)

    robj.fit(data_train,target_train)
    print("\nAccuracy of Training Data using Random Forest Classifier is : ",robj.score(data_train,target_train) * 100)
    print("Accuracy of Testing Data using Random Forest Classifier is : ",robj.score(data_test,target_test) * 100)

def bagging_classifier(data_train,data_test,target_train,target_test) :
    print("\n---------------------------Bagging Classifier Alogorithm------------------------------------")

    bobj = BaggingClassifier(DecisionTreeClassifier(),max_samples = 1.0,max_features = 1.0,n_estimators = 100,warm_start = True)

    bobj.fit(data_train,target_train)

    print("Accuracy of Training Data using Bagging Classifier is :",bobj.score(data_train,target_train)*100)
    print("Accuracy of Testing Data using Bagging Classifier is :",bobj.score(data_test,target_test)*100)

def main() :
    print("Jay Ganesh.......")

    Data = pd.read_csv('mnist.csv')

    print(Data.head())

    data = Data.iloc[:,1:]
    target = Data.iloc[:,0]
    
    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5,random_state = 4)

    DecisionTree(data_train,data_test,target_train,target_test)
    RandomForest(data_train,data_test,target_train,target_test)
    bagging_classifier(data_train,data_test,target_train,target_test)
if __name__ == "__main__" :
    main()