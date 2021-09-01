############################################################################################################################################
##
##  Author : Pakshal Shahikant Jain
##  Date : 26/05/2021
##  Program : Machine Learning of Diabetes Data Set Using Four Different MachineLearning Alogorithm(Pipeline Programming Method)
##
############################################################################################################################################

# Import Required Imports from UserDefined Module/Package
from All_Imports import *

class Demo :
    def __init__(self) :
        self.Data_set = pd.read_csv("diabetes.csv")

        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.Data_set.loc[:,self.Data_set.columns != 'Outcome'],self.Data_set['Outcome'],stratify = self.Data_set['Outcome'],random_state = 66)

    def Display(self) :
        print("--------Inoformation of Diabetics Dataset-----------------")
        print("\n------Columns of Dataset-----------")
        print(self.Data_set.columns)

        print("\n------First 5 Records of dataset--------")
        print(self.Data_set.head())

        print("\nDimension of Diabetes data : {}".format(self.Data_set.shape))

    def DecisionTree(self) :
        print("------Diabetics Predictor using Decision Tree Classifier--------")
        tree = DecisionTreeClassifier(random_state = 0)
        tree.fit(self.X_train,self.Y_train)

        print("Accuracy on training set : {:.3f}".format(tree.score(self.X_train,self.Y_train)*100))

        print("Accuracy on test set : {:.3f}".format(tree.score(self.X_test,self.Y_test)*100))

        tree1 = DecisionTreeClassifier(max_depth = 3,random_state = 0)
        tree.fit(self.X_train,self.Y_train)

        print("Accuracy on training set : {:.3f}".format(tree.score(self.X_train,self.Y_train)*100))
        print("Accuracy on test set : {:.3f}".format(tree.score(self.X_test,self.Y_test)*100))

        print("Feature importance :\n{}".format(tree.feature_importances_))

        plt.figure(figsize=(8,6))
        n_features = 8
        plt.barh(range(n_features),tree.feature_importances_,align  = 'center')
        diabetes_features = [x for i,x in enumerate(self.Data_set.columns) if i != 8]
        plt.yticks(np.arange(n_features),diabetes_features)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1,n_features)
        plt.show()

    def KNN(self) :
        print("\n------Diabetics Predictor using KNN Alogorithm--------\n")
        training_accuracy = [] 
        test_accuracy = [] 

        neighbors_settings = range(1,11)

        for n_neighbors in neighbors_settings :
            knn = KNeighborsClassifier(n_neighbors = n_neighbors)
            knn.fit(self.X_train,self.Y_train)

            training_accuracy.append(knn.score(self.X_train,self.Y_train))

            test_accuracy.append(knn.score(self.X_test,self.Y_test))
        
        plt.plot(neighbors_settings,training_accuracy,label = "training accuracy")
        plt.plot(neighbors_settings,test_accuracy,label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighors")
        plt.legend()
        plt.savefig('Knn_compare_model')
        plt.show()

        knn = KNeighborsClassifier(n_neighbors = 9)
        knn.fit(self.X_train,self.Y_train)

        print("Accuracy of K-NN classifier on training set : {:.2f}%".format(knn.score(self.X_train,self.Y_train)*100,"%"))

        print("Accuracy of KNN classifier on test set : {:.2f}%".format(knn.score(self.X_test,self.Y_test)*100))        

    def LogReg(self) :
        print("\n------Diabetics Predictor using Logistic Regression--------\n")

        logreg = LogisticRegression(max_iter = 2000).fit(self.X_train,self.Y_train)

        print("Training set Accuracy : {:.3f}%".format(logreg.score(self.X_train,self.Y_train)*100))

        print("Test set Accuracy :{:.3f}%".format(logreg.score(self.X_test,self.Y_test)*100))

        logreg001 = LogisticRegression(C = 0.01,max_iter = 2000).fit(self.X_train,self.Y_train)

        print("Training set accuracy : {:.3f}%".format(logreg001.score(self.X_train,self.Y_train)*100))
        print("Test set Accuracy :{:.3f}%".format(logreg001.score(self.X_test,self.Y_test)*100))

    def Rand_Forest(self) :
        print("\n---------------Diabetes predictor using Random Forest---------------\n")
        rf = RandomForestClassifier(n_estimators = 100,random_state = 0)
        rf.fit(self.X_train,self.Y_train)
        print("Accuracy on training set : {:.3f}%".format(rf.score(self.X_train,self.Y_train)*100))
        print("Accuracy on test set :{:.3f}%".format(rf.score(self.X_test,self.Y_test)*100))

        rf1 = RandomForestClassifier(max_depth = 3,n_estimators = 100,random_state = 0)
        rf1.fit(self.X_train,self.Y_train)
        print("Accuracy on training set :{:.3f}%".format(rf1.score(self.X_train,self.Y_train)*100))
        print("Accuracy on testing set : {:.3f}%".format(rf1.score(self.X_test,self.Y_test)*100))

        plt.figure(figsize = (8,6))
        n_features = 8 
        plt.barh(range(n_features),rf.feature_importances_,align = 'center')
        diabetes_features = [x for i,x in enumerate(self.Data_set.columns) if i != 8]
        plt.yticks(np.arange(n_features),diabetes_features)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1,n_features)
        plt.show()

def main() :
    print("Jay Ganesh.................")

    dobj = Demo()

    dobj.Display()
    dobj.DecisionTree()
    dobj.KNN()
    dobj.LogReg()
    dobj.Rand_Forest()

if __name__ == "__main__" :
    main()