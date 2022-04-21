import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

 
class SVM :

    def __init__(self, C=1.0):        # C is the hyperparameter "Regularization Constant" that determines to what extent the soft margine would be

        self.C = C
        self.B = None
        self.b = None
    
    def hyperplane(self,X):
        return X.dot(self.B) + self.b
 
    def fit(self, X, y, LR=0.001, iterations=500):

        # Initialize  hyperplane equation parameters B(beta) and b(bias)
        space_dimension=X.shape[1]
        self.B = np.random.randn(space_dimension)      # B is the beta in hyperplane equation [h(x)=B1*X1+B2*X2+....+b] has number of values according to number of features
        self.b = 0                                     # b is the bias in previous equation 
        
        #cost_arr = []

        for i in range(iterations):
            decision=X.dot(self.B) + self.b
            self.sign=np.sign(decision)
            margin = y * decision                # margine equation y(B1*X1+B2*X2+....+B)=1&-1 according to the label 1 &-1  ==== or zero in data point lies on that plane 
          
           # Gradient descent
            #cost= 0.5* self.B.dot(self.B) + self.C * np.sum(np.maximum(0, 1 - margin))     # cost function
            #cost_arr.append(cost)
            #print(cost)
 
            wrong_class = np.where(margin < 1)[0]
            d_B = self.B - self.C * y[wrong_class].dot(X[wrong_class])         # derivative of cost function to beta
            self.B = self.B - LR * d_B                                         # updated beta
            d_b = - self.C * np.sum(y[wrong_class])                            # derivative of cost function to bias
            self.b = self.b - LR * d_b                                         # updated bias
        self.support_vectors_train = np.where(margin <= 1)[0]                  # used in plotting




         
 
    def plot(self,X,y,k):

        if k == 0:
            support_vectors=self.support_vectors_train
        else:
            support_vectors=self.support_vectors_test

        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Paired, alpha=.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
 
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.hyperplane(xy).reshape(XX.shape)
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])
        # highlight the support vectors
        ax.scatter(X[:, 0][support_vectors], X[:, 1][support_vectors], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
        plt.show() 
 
    def predict(self,X,y):

       prediction= np.sign(self.hyperplane(X))
       margin_test = y * self.hyperplane(X)     # for plot
       self.support_vectors_test = np.where(margin_test <= 1)[0]  # for plot
       return np.mean(y ==  prediction)
 
    
 
 
if __name__ == '__main__':

    #=========================================Loading and preparing the data for binary classification in 2d =====================

    data= sns.load_dataset("iris")   # columns : sepal_length,sepal_width,petal_length,petal_width,species
    #print(data.shape)  # (150,5)
    data= data.tail(100)             # to use the data in binary classification: use only last 100 points as they contain only two classes
    #print(data.shape)  #(100,5)
    encode = preprocessing.LabelEncoder()
    labels= encode.fit_transform(data["species"])
    labels[labels == 0] = -1         # replacing 0 labels with -1 to work with the equations that is based on that
    #print(labels.shape)             # (100,) only 1&0
    data= data.drop(["species"], axis=1)
    data=data.iloc[:,2:4]
    data=np.asarray(data)
    # preprocessing: Standardize the data.
    scale= StandardScaler()
    data= scale.fit_transform(data)

    x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.4,random_state=42)

    #========================================== using built in classifier to check=========================================
    sklearn_svm = svm.SVC(C = 15, kernel='linear')
    sklearn_svm.fit(x_train, y_train)
    print('sklearn accuracy using svm',sklearn_svm.score(x_test, y_test))

    #==================================================================================================================================
    
    # Running the model  
    model = SVM(C=15.0)
    model.fit(x_train, y_train)
    print("accuaracy of implemented algorithm:", model.predict(x_test,y_test))

    # to plot with train data use X_train and 0 & X_test and 1 to plot test data 
    model.plot(x_train,y_train,0)
