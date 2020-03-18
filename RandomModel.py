import numpy as np
import PlotEvaluationMetrics
import Functions


metaLibraryKey = Functions.ask_user_to_input_metaLibraryKey()
if metaLibraryKey == 743:
    n_test = 216
elif metaLibraryKey == 39:
    n_test = 110
elif metaLibraryKey == 454:
    n_test = 96
else:
    n_test = 88

anomaly_indices = Functions.get_indices(metaLibraryKey)
n_repeats = 1
anomalies = np.zeros(n_test)
for i in anomaly_indices:
    anomalies[i] = 1

candidate_values = np.linspace(0.01, 0.5, 1000)
max_F1_scores = list()
max_Recall_scores = list()
max_Precision_scores = list()
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

