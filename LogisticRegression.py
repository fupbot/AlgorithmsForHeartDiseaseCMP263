#Final Project for CMP263 - Machine Learning at UFRGS 2022 - Porto Alegre, Brazil
#Code for Logistic Regression implementation of UCI Heart Disease Data
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
x = dataset.iloc[:, :-1].values               #Set X to independent variables
y = dataset.iloc[:, -1].values                #Set Y to dependent variable

#Create label encoder to convert classes into numerical form
LabelEncoder = preprocessing.OneHotEncoder()

#Convert string to numbers
x_encoded=LabelEncoder.fit_transform(x).todense() #a dense np matrix is needed for model fitting

#Import specific sublibraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

#A function is created for training and testing the Logistic Regression multiple times
def log_reg(C_log):
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
    
    #Creating the logistic regression model
    #Solver is liblinear because it is the most appropriate for binary cases
    #C is regularization parameter and it is varied according to the function call
    model = LogisticRegression(solver='liblinear', C = C_log, multi_class='ovr')
    model.fit(x_train, y_train)
    
    #Evaluation Metriccs
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test)).ravel()
    log_stats = classification_report(y_test, model.predict(x_test), output_dict = True)
    reg_log_stats = []                #Accuracy, TP_rate, FP_rate, Precision, Recall

    #Returns a dictionary with evaluation metrics
    reg_log_stats.append(model.score(x_test, y_test))
    reg_log_stats.append(tp/(tp+fn))
    reg_log_stats.append(fp/(fp+tn))
    reg_log_stats.append(log_stats['buff']['precision'])
    reg_log_stats.append(log_stats['buff']['recall'])
    
    return reg_log_stats

#Variables for evaluation metrics
accuracy = 0.0
TP_rate = 0.0
FP_rate = 0.0
precision = 0.0
recall = 0.0
stats = []
test_num = 200     #number of repetitions

for i in range(0, test_num):
    stats = log_reg(C_log = 0.03)
    accuracy = accuracy + stats[0]
    TP_rate = TP_rate + stats[1]
    FP_rate = FP_rate + stats[2]
    precision = precision + stats[3]
    recall =  recall + stats[4]
    
#Main Statistics
print("Statistics for Logistic Regression:")
print("Accuracy: ", accuracy/test_num)
print("TP rate: ", TP_rate/test_num)
print("FP rate: ", FP_rate/test_num)
print("Precision: ", precision/test_num)
print("Recall: ",recall/test_num)


#Code for varying and analyzing influence of C
accuracy1 = 0.0
TP_rate1 = 0.0
FP_rate1 = 0.0
precision1 = 0.0
recall1 = 0.0
stats1 = []
params1 = {}
test_num1 = 10
max_c = 1.0    #default of sklearn
evolution_stats = []

c_array = np.linspace(0.001, max_c, 50)

for i in range(0, len(c_array)):
    for j in range(0, test_num1):
        stats1 = log_reg(C_log = c_array[i])
        accuracy1 = accuracy1 + stats1[0]
        TP_rate1 = TP_rate1 + stats1[1]
        FP_rate1 = FP_rate1 + stats1[2]
        precision1 = precision1 + stats1[3]
        recall1 =  recall1 + stats1[4]
    evolution_stats.append([accuracy1/test_num1, TP_rate1/test_num1, FP_rate1/test_num1, precision1/test_num1, recall1/test_num1])
    accuracy1, TP_rate1, FP_rate1, precision1, recall1 = 0.0,0.0,0.0,0.0,0.0

acc = []
tp_rate = []
fp_rate = []
prec = []
rec = []
for element in evolution_stats:
    acc.append(element[0])
    tp_rate.append(element[1])
    fp_rate.append(element[2])
    prec.append(element[3])
    rec.append(element[4])
    
plt.figure(figsize=(12, 10))
plt.plot(c_array, acc, 'b')
plt.plot(c_array, tp_rate, 'y')
plt.plot(c_array, fp_rate, 'g')
plt.plot(c_array, prec, 'r')
plt.plot(c_array, rec, 'm')
plt.xlabel('Regularization Value C', fontsize=15)
plt.ylabel('Perfomance Metric', fontsize=15)
plt.legend(['Accuracy', 'TP rate', 'FP rate', 'Precision', 'Recall'],  fontsize=15, loc='lower right')
plt.show()