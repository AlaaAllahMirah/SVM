
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from statistics import mean

data= pd.read_csv('data.txt', delim_whitespace=True,header=None)
#print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())   # no missing values

# Extracting the data
x=data.iloc[:,0:24]   # data without labels
y=data.iloc[:,24]     # labels
#print(x.shape)  #(800,24)
#print(y.shape)  #(800,1)

accuracy=[]
accuracy_norm=[]

clf= svm.SVC(kernel='linear')

for folds in range(10):

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)  # splitting data with 60% train
     #===================================without normalization===========================
    SVM=clf.fit(x_train,y_train)
    prediction=SVM.predict(x_test)
    TN, FP, FN, TP=confusion_matrix(y_test, prediction).ravel()
    acc =  (TP+TN) /(TP+FP+TN+FN)
    accuracy.append(acc)    

   #==================================normalization with same indices================================

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()            # standadization
    x_train_normalized = sc.fit_transform(x_train)
    x_test_normalized = sc.transform(x_test)

    # from sklearn.preprocessing import MinMaxScaler
    # scale = MinMaxScaler()      # normalizatin with built in function
    # x_train_normalized = scale.fit_transform(x_train)
    # x_test_normalized = scale.transform(x_test)
    
    
    #implementation of normalization

    x_train_normalized= np.zeros((x_train.shape[0],x_train.shape[1]))
    x_test_normalized = np.zeros((x_test.shape[0],x_test.shape[1]))
    for j in range(x_train.shape[1]):
        train_feature=x_train.iloc[:,j] 
        norm_train_feature = (train_feature - np.min( train_feature)) / (np.max( train_feature)-np.min( train_feature))
        x_train_normalized[:,j] = norm_train_feature
        test_feature=x_test.iloc[:,j] 
        norm_test_feature=(test_feature - np.min( train_feature)) / (np.max( train_feature)-np.min( train_feature))
        x_test_normalized [:, j] = norm_test_feature    
#====================================normalized acuuracies========================
    SVM_normalized=clf.fit(x_train_normalized,y_train)
    prediction_norm=SVM.predict(x_test_normalized)
    TN_norm, FP_norm, FN_norm, TP_norm=confusion_matrix(y_test, prediction_norm).ravel()
    acc_norm =  (TP_norm+TN_norm) /(TP_norm+FP_norm+TN_norm+FN_norm)
    accuracy_norm.append(acc_norm)


print("Average accuracy without normalization :",sum(accuracy) / len(accuracy))
print("Average accuracy with normalization:",sum(accuracy_norm) / len(accuracy_norm))

   







