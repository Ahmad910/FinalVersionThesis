from scipy.stats import sem, t
import numpy as np

file = open('means.txt', 'r')
scores = file.readlines()
max_F1_scores = list()
n_repeats = 50
for score in scores:
    score = score.split()
    max_F1_scores.append(float(score[0]))
confidence = 0.95
print("higheeeeeest: ", max(max_F1_scores))
max_F1_scores = 1.0 * np.array(max_F1_scores)
mean_F1_score = np.mean(max_F1_scores)
std_err = max_F1_scores.std()
h = std_err * t.ppf((1 + confidence) / 2., n_repeats - 1)
start = mean_F1_score - h
end = mean_F1_score + h
print("The mean is: ", mean_F1_score)
print("The confidence interlval is: [", start, ", ", end, "]")