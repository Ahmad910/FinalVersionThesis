import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PlotEvaluationMetrics
#n_test = 216 #743
#n_test = 96 #454
n_test = 110 #39
#n_test = 88 #40

# anomaly_indices = [42, 43, 44, 45, 51, 74, 102, 206]  # 743 Malmö Stad
anomaly_indices = [4, 20, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72] # 39FD
#anomaly_indices = list(range(24, 84))# 40FD
#anomaly_indices = list(range(65, 96))  # 454 MS
n_repeats = 1
anomalies = np.zeros(n_test)
for i in anomaly_indices:
    anomalies[i] = 1

candidate_values = np.linspace(0.01, 0.5, 1000)
max_F1_scores = list()
max_Recall_scores = list()
max_Precision_scores = list()
print(np.linspace(0.01, 2, 10))
print(np.random.randint(0, 2, n_test))
for i in range(n_repeats):
    f, r, p = PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, np.random.randint(0, 2, n_test), 0)
    max_F1_scores.append(f)
    max_Recall_scores.append(r)
    max_Precision_scores.append(p)

max_F1_scores = np.asarray(max_F1_scores)
max_Recall_scores = np.asarray(max_Recall_scores)
max_Precision_scores = np.asarray(max_Precision_scores)

print('Mean of best F1: ', max_F1_scores.mean())
print('Mean of best Recall: ', max_Recall_scores.mean())
print('Mean of best precision: ',  max_Precision_scores.mean())

