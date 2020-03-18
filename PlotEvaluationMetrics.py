import matplotlib.pyplot as plt
import numpy as np

def find_best_theta(candidate_values, actual_anomalies, errors):
    recall_values = list()
    precision_values = list()
    F1__values = list()
    for theta in candidate_values:
        predicted_anomalies = get_predictions(theta,errors)
        recall = calculate_recall(actual_anomalies, predicted_anomalies)
        precision = calculate_precision(actual_anomalies, predicted_anomalies)
        F1 = calculate_F1(actual_anomalies, predicted_anomalies)
        recall_values.append(recall)
        precision_values.append(precision)
        F1__values.append(F1)
    plt.plot(candidate_values, np.asarray(recall_values), color='b', label='recall')
    plt.plot(candidate_values, np.asarray(precision_values), color='c', label='precision')
    plt.plot(candidate_values, np.asarray(F1__values), color='m', label='F1-score')
    plt.legend()
    plt.show()
    return max(F1__values), max(recall_values), max(precision_values)

def get_predictions(theta, errors):
    predicted_anomalies = list()
    for i in range(len(errors)):
        if errors[i] > theta:
            predicted_anomalies.append(1)
        else:
            predicted_anomalies.append(0)
    return np.asarray(predicted_anomalies)

def calculate_test(actual_anomalies, predicted_anomalies):
    true_pos = 0
    false_neg = 0
    for i in range(len(actual_anomalies)):
        if actual_anomalies[i] == 1:
            if predicted_anomalies[i] == 1:
                true_pos += 1
            else:
                false_neg += 1
    return true_pos, false_neg

#Precision = true positives / (true postitives + false positives)
def calculate_recall(actual_anomalies, predicted_anomalies):
    true_pos= 0
    false_neg = 0
    for i in range(len(actual_anomalies)):
        if actual_anomalies[i] == 1:
            if predicted_anomalies[i] == 1:
                true_pos += 1
            else:
                false_neg +=1
    return true_pos / (true_pos + false_neg)

def calculate_precision(actual_anomalies, predicted_anomalies):
    true_pos = 0
    false_pos = 0
    for i in range(len(predicted_anomalies)):
        if predicted_anomalies[i] == 1:
            if actual_anomalies[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
    return true_pos / (true_pos + false_pos)

def calculate_F1(actual_anomalies, predicted_anomalies):
    recall = calculate_recall(actual_anomalies, predicted_anomalies)
    precision = calculate_precision(actual_anomalies, predicted_anomalies)
    if recall + precision == 0:
        return 0
    else:
        return (2*((recall*precision)/(recall + precision)))

