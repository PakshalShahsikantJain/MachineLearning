#####################################################################################################
##
##  Author : Pakshal Shashikant Jain
##  Date : 14/04/2021
##  Program : Machine Learning of Advertising DataSet using Linear Regression Algorithm 
##
####################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class MarvellousLinearRegression :
    def __init__(self,Data) :
        self.data = Data

    def Mean(self,Data) :
        sum = 0
        for i in range(len(Data)) :
            sum = sum + Data[i]
        
        #print("Value of Sum is :",sum)
        return sum / len(Data)

    def MarvellousLRTV(self) :
        print("----------------------------------MarvellousLRTV---------------------------------------------")
        X = self.data["TV"].values
        Y = self.data["sales"].values 

        Mean_X = self.Mean(X)
        Mean_Y = self.Mean(Y)

        print("Value of XB is :",Mean_X)
        print("Value of YB is :",Mean_Y)

        Numerator = 0
        Denominator = 0

        # slope(m) = sum((x - xb) * (y - yb)) / sum(x - xb)^2

        for i in range(len(X)) :
            Numerator = Numerator + (X[i] - Mean_X)*(Y[i] - Mean_Y)
            Denominator = Denominator + (X[i] - Mean_X)**2

        print("Value of Numerator is :",Numerator)
        print("Value of Denominator is :",Denominator)

        slope_m = Numerator / Denominator
        print("Value of Slope is :",slope_m)

        # yintercept(c) = Yb - (m * Xb)

        c = Mean_Y - (slope_m * Mean_X)
        print("Value of y intercept is :",c)

        #Rsquare = sum(yp - YB)^2 / sum(Y - YB)^2

        yp = []
        for i in range(len(Y)) :
            YP = slope_m * X[i] + c 
            yp.append(YP)
        
        Rsquare = 0
        Numerator = 0
        Denominator = 0

        for i in range(len(Y)) :
            Numerator = Numerator + (yp[i] - Mean_Y)**2
            Denominator = Denominator + (Y[i] - Mean_Y)**2
        
        Rsquare = Numerator / Denominator
        print("Values of Rsquare is :",Rsquare)

        X_Start = np.min(X)
        X_End = np.max(X)

        x = np.linspace(X_Start,X_End)
        y = slope_m*x + c

        plt.plot(x,y,color = 'r',label = "Line of Regression")
        plt.scatter(X,Y,color = 'b',label = "Data Plot")

        plt.xlabel("TV")
        plt.ylabel("Sales")

        plt.legend()
        plt.show()

    def MarvellousLRR(self) :
        print("\n----------------------------------MarvellousLRR---------------------------------------------")
        
        X = self.data["radio"].values
        Y = self.data["sales"].values

        Mean_X = self.Mean(X)
        Mean_Y = self.Mean(Y)
        print("Value of Xbar is :",Mean_X)
        print("Value of YBar is :",Mean_Y)

        #slope(m) = sum((x - XB)*(y - YB)) / sum(x - XB)^2  
        m = 0
        Numerator = 0
        Denominator = 0

        for i in range(len(X)) :
            Numerator = Numerator + (X[i] - Mean_X)*(Y[i] - Mean_Y)
            Denominator = Denominator + (X[i] - Mean_X)**2
        
        print("Value of Numeartor is :",Numerator)
        print("Value of Denominator is :",Denominator)

        m = Numerator / Denominator
        print("Value of Slope is :",m)

        #c = YB - (m * XB)
        c = Mean_Y - (m * Mean_X)
        print("Value of Y intercept is :",c)

        yp = []
        for i in range(len(Y)) :
            YP = m * X[i] + c 
            yp.append(YP)

        Rsquare = 0
        Numerator = 0
        Denominator = 0

        for i in range(len(Y)) :
            Numerator = Numerator + (yp[i] - Mean_Y)**2
            Denominator = Denominator + (Y[i] - Mean_Y)**2
        
        Rsquare = Numerator / Denominator
        print("Values of Rsquare is :",Rsquare)
        
        X_start = np.min(X)
        X_end = np.max(X)

        x = np.linspace(X_start,X_end)
        y = m*x + c

        plt.plot(x,y,color = 'r',label = "Line of Regression")
        plt.scatter(X,Y,color = 'b',label = "Data Plot")

        plt.xlabel("Radio")
        plt.ylabel("Sales")

        plt.legend()
        plt.show()

    def MarvellousLRNP(self) :
        print("\n----------------------------------MarvellousLRNP---------------------------------------------")

        X = self.data["newspaper"].values
        Y = self.data["sales"].values

        Mean_X = self.Mean(X)
        Mean_Y = self.Mean(Y)
        print("Value of Xbar is :",Mean_X)
        print("Value of YBar is :",Mean_Y)

        #slope(m) = sum((x - XB)*(y - YB)) / sum(x - XB)^2  
        m = 0
        Numerator = 0
        Denominator = 0

        for i in range(len(X)) :
            Numerator = Numerator + (X[i] - Mean_X)*(Y[i] - Mean_Y)
            Denominator = Denominator + (X[i] - Mean_X)**2
        
        print("Value of Numeartor is :",Numerator)
        print("Value of Denominator is :",Denominator)

        m = Numerator / Denominator
        print("Value of Slope is :",m)

        #c = YB - (m * XB)
        c = Mean_Y - (m * Mean_X)
        print("Value of Y intercept is :",c)

        yp = []
        for i in range(len(Y)) :
            YP = m * X[i] + c 
            yp.append(YP)

        Rsquare = 0
        Numerator = 0
        Denominator = 0

        for i in range(len(Y)) :
            Numerator = Numerator + (yp[i] - Mean_Y)**2
            Denominator = Denominator + (Y[i] - Mean_Y)**2
        
        Rsquare = Numerator / Denominator
        print("Values of Rsquare is :",Rsquare)
        
        X_start = np.min(X)
        X_end = np.max(X)

        x = np.linspace(X_start,X_end)
        y = m*x + c

        plt.plot(x,y,color = 'r',label = "Line of Regression")
        plt.scatter(X,Y,color = 'b',label = "Data Plot")

        plt.xlabel("newspaper")
        plt.ylabel("Sales")

        plt.legend()
        plt.show()

def main() :
    print("Jay Ganesh.........")
    print("Enter Name of csv file You Want To Read")
    name = input() 

    data = pd.read_csv(name)
    #print(len(data))
    obj = MarvellousLinearRegression(data)
    obj.MarvellousLRTV()
    obj.MarvellousLRR()
    obj.MarvellousLRNP()

if __name__ == "__main__" :
    main()