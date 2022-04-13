#Final Project for CMP263 - Machine Learning at UFRGS 2022 - Porto Alegre, Brazil
#Code for KNN and RandomForests implementation of UCI Heart Disease Data
#Paper "Comparative Analysis of Classification Algorithms for the Prediction of Heart Disease"
#Authors Arthur Medeiros, Fabio Pereira e Gustavo Loguercio

import math, operator, random, copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

random.seed(263) # from CMP-263 :D
dataset = []

MIN_FLOAT = 1.17549e-038

MAX_FLOAT = 3.40282e+038

with open("dados_coracao_ML.csv", "r") as f:
    for line in f:
        newline = line.split("\n")[0].split(",")
        dataset.append(newline)
        

# Numeric attributes get normalized
is_numeric = [True, False, False, True, True, False, False, True, False, True, False, True, False, False]

# Pre Processing
# Get minimum and maximum values for the numeric variables (may be used later for normalization)
min_values = []
max_values = []

for attr in dataset[0]:
    min_values.append(MAX_FLOAT)
    max_values.append(MIN_FLOAT)
        
        
for instance in dataset: 
    i = 0
    for attr in instance:
        if (is_numeric[i]):
            min_values[i] = min(float(instance[i]), min_values[i])
            max_values[i] = max(float(instance[i]), max_values[i])
        i += 1


# This function will normalize every numeric value to a 0-1 float range
def normalize_numeric(value, min, max):
    return (value-min)/(max-min)

normalized_dataset = []

for instance in dataset:
    normalized_instance = []
    # We are about to normalize the dataset for proper distance measurement now
    # We are going to map every feature to the 0-1 range

    # Age is normalized considering the 0-100 range
    normalized_instance.append(float(instance[0])/ 100)
    
    # Male = 0, Female = 1
    if (instance[1] == "male"):
        normalized_instance.append(float(0))
    elif (instance[1] == "fem"):
        normalized_instance.append(float(1))
    else:
        raise "Invalid Entry"

    # Presence of chest pain:
    # Angina = 0
    # Abnormal Angina = 0.33
    # Non-Anginal = 0.66
    # Asymptomatic = 1
    if (instance[2] == "angina"):
        normalized_instance.append(float(0))
    elif (instance[2] == "abnang"):
        normalized_instance.append(1/3)
    elif (instance[2] == "notang"):
        normalized_instance.append(2/3)
    elif (instance[2] == "asympt"):
        normalized_instance.append(1)
    else:
        raise "Invalid Entry"

    # Normalize resting blood pressure
    normalized_value = normalize_numeric(float(instance[3]), min_values[3], max_values[3])
    normalized_instance.append(normalized_value)

    # Normalize cholesterol
    normalized_value = normalize_numeric(float(instance[4]), min_values[4], max_values[4])
    normalized_instance.append(normalized_value)

    # If fasting blood sugar is higher than 120 or not. True = 1, False = 0
    if (instance[5] == "true"):
        normalized_instance.append(float(1))
    elif (instance[5] == "fal"):
        normalized_instance.append(float(0))
    else:
        raise "Invalid Entry"

    # Resting ECG test result. Normal = 0, abnormal = 0.5, Hyper = 1
    if (instance[6] == "norm"):
        normalized_instance.append(float(0))
    elif (instance[6] == "abn"):
        normalized_instance.append(float(0.5))   
    elif (instance[6] == "hyp"):
        normalized_instance.append(float(1))
    else:
        raise "Invalid Entry"   

    # Normalize Max Heart Rate
    normalized_value = normalize_numeric(float(instance[7]), min_values[7], max_values[7])
    normalized_instance.append(normalized_value)

    # Presence of exercise-induced angina (True = 1, False = 0)
    if (instance[8] == "true"):
        normalized_instance.append(float(1))
    elif (instance[8] == "fal"):
        normalized_instance.append(float(0))
    else:
        raise "Invalid Entry"


    # Normalize Oldpeak
    normalized_value = normalize_numeric(float(instance[9]), min_values[9], max_values[9])
    normalized_instance.append(normalized_value)

    # Slope direction. Up = 0, Flat = 0.5, Down = 1
    if (instance[10] == "up"):
        normalized_instance.append(float(0))
    elif (instance[10] == "flat"):
        normalized_instance.append(float(0.5))
    elif (instance[10] == "down"):
        normalized_instance.append(float(1))
    else:
        raise "Invalid Entry"

    # Numbers of colored vessels, normalized
    normalized_value = normalize_numeric(float(instance[11]), min_values[11], max_values[11])
    normalized_instance.append(normalized_value)

    # Thalassemia (a blood disorder): Normal = 0, Fixed = 0.5, reversable effect = 1
    if (instance[12] == "norm"):
        normalized_instance.append(float(0))
    elif (instance[12] == "fix"):
        normalized_instance.append(float(0.5))
    elif (instance[12] == "rev"):
        normalized_instance.append(float(1))
    else:
        raise "Invalid Entry"

    # What we are trying to predict. Buff/Healthy = 0, Sick = 1    
    if (instance[13] == "buff"):
        normalized_instance.append(float(0))
    elif (instance[13] == "sick"):
        normalized_instance.append(float(1))
    else:
        raise "Invalid Entry"

    normalized_dataset.append(normalized_instance)

# These will be used for plots, they are the according metrics
accuracy_pltpoints = []         # Accuracy
buff_recall_pltpoints = []      # Buff class recall
buff_precision_pltpoints = []   # Buff class precision
sick_recall_pltpoints = []      # Sick class recall
sick_precision_pltpoints = []   # Sick class precision

max_k = 25  
for k in range(1, max_k):
    print("K: ", k)
    # Counts how many true positives and false positives for each class
    metrics = [[0,0], [0,0]] # metrics[real][predicted]

    num_of_iterations = 200
    # Will iterate num_of_iterations times to have more stable metrics
    for iteration in range (0, num_of_iterations):
        if (iteration % 10 == 0):
            print("Iteration: ", iteration)
        temp_dataset = copy.deepcopy(normalized_dataset)
        training_dataset = []
        testing_dataset = []

        total_size = len(temp_dataset)

        # The algorithm will split the dataset according to this ratio
        training_split = 0.7
        testing_split = 0.3

        training_size = math.floor(training_split * total_size)
        testing_size = total_size - training_size

        for i in range(0, testing_size):
            remaining_size = total_size - i
            chosen_index = math.floor(random.random() * remaining_size)
            testing_dataset.append(temp_dataset[chosen_index])
            temp_dataset.remove(temp_dataset[chosen_index])

        training_dataset = temp_dataset

        # Separate prediction labels to different lists (requirement for scikit-learn)
        training_labels = []
        testing_labels = []

        for training_instance in training_dataset:
            training_labels.append(training_instance.pop(-1))

        for testing_instance in testing_dataset:
            testing_labels.append(testing_instance.pop(-1))

        # Uncomment the following lines if you want to use KNN:

        #knn = KNeighborsClassifier(n_neighbors=k)
        #knn.fit(training_dataset, training_labels)
        #prediction = knn.predict(testing_dataset)

        # Uncomment the following lines if you want to use Random Forest:

        clf = RandomForestClassifier(max_depth = k)
        clf.fit(training_dataset, training_labels)
        prediction = clf.predict(testing_dataset)

        correct = 0
        for i in range(0, testing_size):
            if prediction[i] == testing_labels[i]:
                correct += 1
            if (prediction[i] != 1 and prediction[i] != 0):
                raise "Something is wrong"
            metrics[int(testing_labels[i])][int(prediction[i])] += 1

    # Calculate all metrics
    total = metrics[0][0] + metrics[1][1] + metrics[0][1] + metrics[1][0]
    correct = metrics[0][0] + metrics[1][1]
    accuracy = correct/total
    buff_recall = metrics[0][0] / (metrics[0][0] + metrics[0][1])
    buff_precision = metrics[0][0] / (metrics[0][0] + metrics[1][0])
    sick_recall = metrics[1][1] / (metrics[1][0] + metrics[1][1])
    sick_precision = metrics[1][1] / (metrics[0][1] + metrics[1][1])

    # Add the metrics to the list (that may be plotted)
    accuracy_pltpoints.append(accuracy)
    buff_recall_pltpoints.append(buff_recall)
    buff_precision_pltpoints.append(buff_precision)
    sick_recall_pltpoints.append(sick_recall)
    sick_precision_pltpoints.append(sick_precision)
    print("\n")

# X axis
k_ranges = list(range(1,max_k))

#plt.xlabel('Neighborhood Size (k)')

#plt.plot(k_ranges, accuracy_pltpoints, label = "Accuracy")
#plt.ylim([0.5, 1])
#plt.xlabel('Neighborhood Size (k)')

#plt.ylabel('Accuracy')

#plt.title("Neighborhood size effect on accuracy")

#plt.show()

#plt.clf()


#plt.plot(k_ranges, buff_recall_pltpoints, label = "'Healthy' Recall")
#plt.plot(k_ranges, buff_precision_pltpoints, label = "'Healthy' Precision")
#plt.ylim([0.5, 1])
#plt.xlabel('Neighborhood Size (k)')

#plt.ylabel('Metric')

#plt.title("Neighborhood size effect on 'Healthy' class metrics")
#plt.legend(loc = "lower right")

#plt.show()
#plt.clf()

#plt.plot(k_ranges, sick_recall_pltpoints, label = "'Sick' Recall")
#plt.plot(k_ranges, sick_precision_pltpoints, label = "'Sick' Precision")
#plt.ylim([0.5, 1])
#plt.xlabel('Neighborhood Size (k)')

#plt.ylabel('Metric')

#plt.title("Neighborhood size effect on 'Sick' class metrics")
#plt.legend(loc = "lower right")

#plt.show()


plt.plot(k_ranges, accuracy_pltpoints, label = "Accuracy")
plt.ylim([0.5, 1])
plt.xlabel('Max depth value')

plt.ylabel('Accuracy')

plt.title("Max tree depth effect on accuracy")

plt.show()

plt.clf()


plt.plot(k_ranges, buff_recall_pltpoints, label = "'Healthy' Recall")
plt.plot(k_ranges, buff_precision_pltpoints, label = "'Healthy' Precision")
plt.ylim([0.5, 1])
plt.xlabel('Max depth value')

plt.ylabel('Metric')

plt.title("Max tree depth effect on 'Healthy' class metrics")
plt.legend(loc = "lower right")

plt.show()

plt.clf()
plt.plot(k_ranges, sick_recall_pltpoints, label = "'Sick' Recall")
plt.plot(k_ranges, sick_precision_pltpoints, label = "'Sick' Precision")
plt.ylim([0.5, 1])
plt.xlabel('Max depth value')

plt.ylabel('Metric')

plt.title("Max tree depth effect on 'Sick' class metrics")
plt.legend(loc = "lower right")

plt.show()

