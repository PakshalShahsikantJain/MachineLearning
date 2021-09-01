###################################################################################################################
##
##  Author : Pakshal Shahikant Jain 
##  Date : 06/04/2021
##  Program : Machine Learning of Play Predictor Data Set using KNN Algorithm
##
######################################################################################################################
from Module import *

def MarvellousKNN(Data,Target,Value,Value2) :
    obj = KNeighborsClassifier(n_neighbors = 3)

    obj.fit(Data,Target)

    ret = obj.predict([[Value,Value2]])

    if ret == 1 :
        return True
    else :
        return False

def main() :
    bret = False

    print("Jay Ganesh.....")
    data = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")
    lobj = LabelEncoder()

    data_Wether = lobj.fit_transform(data.Wether.values)

    data_Temperature = lobj.fit_transform(data.Temperature.values)

    feature = list(zip(data_Wether,data_Temperature))

    data_Play = lobj.fit_transform(data.Play.values)
    label = data_Play

    print("Enter Wether")
    No = int(input())

    print("Enter Temperature")
    No2 = int(input())

    bret = MarvellousKNN(feature,label,No,No2)
    
    if bret == True :
        print("We Can Play")
    else :
        print("We Cannot Play")

if __name__ == "__main__" :
    main()