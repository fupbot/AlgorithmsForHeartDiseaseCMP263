#Final Project for CMP263 - Machine Learning at UFRGS 2022 - Porto Alegre, Brazil
#Code for Naive Bayes implementation of UCI Heart Disease Data
#Paper "Comparative Analysis of Classification Algorithms for the Prediction of Heart Disease"
#Authors Arthur Medeiros, Fabio Pereira e Gustavo Loguercio

#Import libraries needed 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import preprocessing

#Import the dataset - it must be on the same folder as this notebook. Otherwise, address should be corrected.
dataset = pd.read_csv('dados_coracao_ML.csv')
x = dataset.iloc[:, :-1].values                #Set X to independent variables
y = dataset.iloc[:, -1].values                 #Set Y to dependent variable

#Create label encoder to convert classes into numerical form
LabelEncoder = preprocessing.OneHotEncoder()

#Convert string to numbers
x_encoded=LabelEncoder.fit_transform(x).todense() #a dense np matrix is needed for model fitting

#Import specific sublibraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#A function is created for training and testing Naive Gaussian Bayes multiple times
def gaussian_Bayes():
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.3) #70% training and 30% testing
    
    #conversion to np.array
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    #Scaling the dataset to make every independent variable have the same scale and not influence the results 
    #in a disproportional matter
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    #Create and train a Gaussian Naive Bayes instance
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    #Predict the outcome using the test part of the dataset
    y_pred = gnb.predict(x_test)
    
    #Array to store statistics
    bayes_stats = []   #Accuracy, TP_rate, FP_rate, Precision, Recall
    
    #Evaluation Metriccs
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    bayes_stats.append(metrics.accuracy_score(y_test, y_pred))
    bayes_stats.append(tp/(tp+fn))
    bayes_stats.append(fp/(fp+tn))
    bayes_stats.append(metrics.precision_score(y_test, y_pred, pos_label = 'buff'))
    bayes_stats.append(metrics.recall_score(y_test, y_pred, pos_label = 'buff'))
           
    #Return bayes statistics    
    return bayes_stats 

#Variables for evaluation metrics
accuracy = 0.0
TP_rate = 0.0
FP_rate = 0.0
precision = 0.0
recall = 0.0
stats = []
test_num = 200    #number of repetitions

for i in range(0, test_num):
    stats = gaussian_Bayes()
    accuracy = accuracy + stats[0]
    TP_rate = TP_rate + stats[1]
    FP_rate = FP_rate + stats[2]
    precision = precision + stats[3]
    recall =  recall + stats[4]
    
#Main Statistics
print("Statistics for Naive Bayes:")
print("Accuracy: ", accuracy/test_num)
print("TP rate: ", TP_rate/test_num)
print("FP rate: ", FP_rate/test_num)
print("Precision: ", precision/test_num)
print("Recall: ",recall/test_num)

