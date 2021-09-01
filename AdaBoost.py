########################################################################################################################
##
##  Author : Pakshal Shashikant Jain
##  Date : 04/06/2021
##  Program : Machine Learning of MNIST Data Set using AdaBoost Classifier 
##
#########################################################################################################################


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier

def AdaBoost(data_train,data_test,target_train,target_test) :
    dobj = DecisionTreeClassifier(splitter = 'random',max_depth = None,min_samples_split = 3,random_state = 4)

    aobj = AdaBoostClassifier(dobj,n_estimators = 150,learning_rate = 1)
    aobj.fit(data_train,target_train)

    print("Accuracy of Training Data set is : ",aobj.score(data_train,target_train)*100)
    print("Accuracy of Testing Data set is : ",aobj.score(data_test,target_test)*100)

def main() :
    print("Jay Ganesh......")
    print("---------------------Machine Learning using AdaboostClassifier Algorithm---------------------")
    Data = pd.read_csv('mnist.csv')

    data = Data.iloc[:,1:]
    target = Data.iloc[:,0]

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.2,random_state = 4)

    AdaBoost(data_train,data_test,target_train,target_test)

if __name__ == "__main__" :
    main()