###################################################################################
#
# Author : Pakshal Shashikant Jain
# Date : 18-06-2021
# Program : Machine Learning of Breast Cancer Case Study using Boosting Classifier 
#
###################################################################################

# Required Pyhton Packages 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pdb
def read_data(path) : 
    """
    Read CSV File of Dataset using Pandas
    :param : dataset:
    :return:
    """

    data = pd.read_csv(path)

    # Data Cleaning 

    data['BareNuclei'] = data['BareNuclei'].replace(['?'],'10')
    
    return data 

########################################################################################

def split_data(data) :
    """
    Split Data Set with test_size
    :param dataset:
    :param No: 
    :return : data_train,data_test,target_train,target_test
    """
    # Split Columns of Fetures and Labels using iloc Function

    # Features of Dataset
    Data = data.iloc[:,0:10] 
    #Labels of Dataset
    target = data.iloc[:,10]

    # Display First Five Records of Features of Dataset
    print("-----------------Features of Dataset-----------------------") 
    print(Data.head())

    # Display First Five Records of Labels of Dataset
    print("------------------Labels of Dataset------------------------")
    print(target.head())

    # Split Dataset for Training and Testing Purpose 

    data_train,data_test,target_train,target_test = train_test_split(Data,target,test_size = 0.3,random_state = 4)

    return data_train,data_test,target_train,target_test

#############################################################################################

def data_statistics(data) :
    """
    Display Basic Statistics of Dataset 
    :param dataset: Pandas DataFrame
    :return:None, print The Basic Statistics of Dataset 
    """
    print(data.describe())

##############################################################################################

def Training(data_train,target_train) :
    dobj = DecisionTreeClassifier(splitter = 'random',max_depth = None,min_samples_split = 3,random_state = 4)
    aobj = AdaBoostClassifier(dobj,n_estimators = 150,learning_rate = 1)

    aobj.fit(data_train,target_train)

    return aobj

##############################################################################################

def main() :
    print("Enter File Name You Want To Read Data From : ")
    path = input()

    data = read_data(path)
    print("-----------------First Five Records of Dataset---------------------")
    print(data.head())

    data_train,data_test,target_train,target_test = split_data(data)

    # Dispaly Value of data_train,data_test,target_train,target_test
    print("----------------------Details of Splitted Data--------------------------")
    print(data_train.head())
    print(target_train.head())
    print(data_test.head())
    print(target_test.head()) 
    
    # Display Basic Statistics of Dataset
    print("--------------------Basic Statistics of Dataset-------------------------")
    data_statistics(data)

    # Display Trained Model
    Trained_Model = Training(data_train,target_train)
    print("Trained Model is :: ",Trained_Model)

    # Testing of Trained Model
    output = Trained_Model.predict(data_test)
    
    for i in range(0,5) :
        print("Actual Outcome :: {} and Predicted Outcome :: {}".format(list(target_test)[i],output[i]))

    print("Training Accuracy is :: ",Trained_Model.score(data_train,target_train) * 100)
    print("Testing Accuracy is :: ",Trained_Model.score(data_test,target_test) * 100)
    print("Confusion Matrix is :: ",confusion_matrix(target_test,output))
if __name__ == "__main__" :
    print("Jay Ganesh........")
    print("-------------Marvellous-------------")
    print("-----Machine Learning of Breast Cancer Case Study using Ada BoostClassifier-----")
    main()