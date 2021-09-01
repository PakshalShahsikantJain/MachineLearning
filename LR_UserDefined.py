###########################################################################################################
##
##  Author : Pakshal Shashikant Jain
##  Date : 11/04/2021
##  Program : Linear Regression Program With All User defined operations without using NumPy Module
##
###########################################################################################################
def MarvellousLinear() :
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    SumX = 0
    SumY = 0
    Xb = 0
    Yb = 0
    
    for i in range(len(X)) :
        SumX = SumX + X[i]
    
    Xb = SumX / len(X)
    #print(Xb)

    for i in range(len(Y)) :
        SumY = SumY + Y[i]
    
    Yb = SumY / len(Y)

    print("We Get Mean as :",Xb,",",Yb)
    
    Numerator = 0
    Denominator = 0
    m = 0
    for i in range(len(X)) :
        Numerator = Numerator + ((X[i] - Xb)*(Y[i] - Yb))
        Denominator = Denominator + (X[i] - Xb)**2
    
    print("We Get Numerator as :",Numerator)
    print("We Get Denominator as :",Denominator)
    m = Numerator / Denominator
    
    print("\nAfter Performing Slope(m) = sum((X-Xb)*(Y-Yb)) / sum(X-Xb)^2 :")
    print("We Get Value of Slope as :",m)

    print("Now To Remove Value of y intercept i.e C we Can Perform C = Yb - m(Xb) : ")
    c = Yb - m * Xb
    print("We Get Value of y intercept as :",c)

    Yp = []
    for i in range(len(Y)) :
        print("Predicted Value of Y i.e Yp is :","Y",i + 1)
        yp = m * X[i] + c
        print(yp)
        Yp.append(yp)
    print(Yp)
    
    Numerator2 = 0
    Denominator2 = 0

    for i in range(len(Yp)) :
        Numerator2 = Numerator2 + (Yp[i] - Yb)**2
        Denominator2 = Denominator2 + (Y[i] -Yb)**2
    
    print("Value of Numerator is :",Numerator2)
    print("Value of Denominator is :",Denominator2)

    print("\nTo Remove Value of RSquare we can perform RSquare = sum(Yp - Yb)^2  / sum(Y - Yb)^2 :")
    RSquare = Numerator2 / Denominator2
    print("We Get Value of RSquare as :",RSquare)

def main() :
    print("-----------Linear Regression-------------------------")

    MarvellousLinear()

if __name__ == "__main__" :
    main()