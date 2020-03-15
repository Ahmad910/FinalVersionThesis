from keras.models import Sequential
from keras import regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
import pandas as pd
from keras.utils import plot_model
from keras.layers import LeakyReLU
from scipy.stats import sem, t
import matplotlib.pyplot as plt
import numpy as np
import PlotEvaluationMetrics
import tensorflow as tf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def build_autoencoder(lookback, n_features):
    model = Sequential()
   # model.add(LSTM(20, activation='tanh', input_shape=(lookback + 1, n_features), activity_regularizer=regularizers.l1(10e-5)))
    model.add(LSTM(4, input_shape=(lookback + 1, n_features)))
    model.add(LeakyReLU(0.3))
    model.add(Dropout(0.75))
    model.add(RepeatVector(lookback + 1))
    #model.add(LSTM(20, activation='tanh', return_sequences=True, activity_regularizer=regularizers.l1(10e-5)))
    model.add(LSTM(4, return_sequences=True))
    model.add(LeakyReLU(0.3))
    model.add(Dropout(0.75))
    model.add(TimeDistributed(Dense(n_features)))
    model.add(LeakyReLU(0.3))
    model.compile(optimizer='adam', loss='mse')
    return model

def transform_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def calculate_error(predicted, actual):
    errors = list()
    for i in range(0, len(predicted)):
        #print(mean_squared_error(predicted[i, :], actual[i, :]))
        errors.append(mean_squared_error(predicted[i, :], actual[i, :]))
    return np.asarray(errors)



df_features = pd.read_csv('autoencoder_743MS.csv')
#df_features = df_features.drop(['ExecutionMonthSin', 'ExecutionMonthCos', 'ExecutionDaySin', 'ExecutionDayCos',
#                                'ExecutionWeekDaySin', 'ExecutionWeekDayCos'], axis=1)
df_features.reset_index(inplace=True, drop=True)

lookback = 30
n_train = 504 #743MS
#n_train = 250 #39FD
#n_train = 228 #40FD
#n_train = 224  #454MS


n_features = df_features.shape[1]

num_obs = df_features.shape[0] - lookback
n_test = num_obs - n_train

data = transform_to_supervised(df_features, n_in=lookback)
data = np.array(data)


x_train = data[:n_train, :]
x_test = data[n_train:, :]
x_train = x_train.reshape((n_train, lookback + 1, n_features))
x_test = x_test.reshape((num_obs - n_train, lookback + 1, n_features))


"""
This was used to print the dataframe and find the index of anomalies
x_test = x_test[:, -1, :]
x_test = x_test.reshape(216, 49)
x_test = pd.DataFrame(x_test)
print(x_test)
"""

n_repeats = 1
error_list = list()
stds = list()
for i in range(n_repeats):
    print("xxxxxxxxxxxxxxxxxxxxxxxxxx: ", i)
    x_test_copy = x_test
    model = build_autoencoder(lookback, n_features)
    history = model.fit(x_train, x_train, epochs=10, batch_size=2, verbose=2, shuffle=False)
    #plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
    predictions = list()
    on_e = 4
    on_b = 2
    for i in range(0, len(x_test), on_b):
        x = x_test_copy[i: i + on_b, :, :]
        if i > 0:
            model.fit(x_test_old, x_test_old, epochs=on_e, batch_size=on_b, shuffle=False, verbose=2)
            #model.reset_states()
        yhat = model.predict(x, batch_size=on_b)
        predictions.append(yhat)
        x_test_old = x
    x_test_copy = x_test_copy[:, -1, :]
    predictions = np.asarray(predictions)
    predictions = predictions.reshape(n_test, lookback + 1, n_features)
    predictions = predictions[:, -1, :]
    errors = calculate_error(predictions, x_test_copy)
    error_list.append(errors)
    stds.append(errors.std())

error_list = np.asarray(error_list)
error_list = error_list.transpose()
mean_errors = list()
for i in range(len(error_list)):
    temp = 0
    for j in range(len(error_list[1])):
        temp += error_list[i][j]
    mean_errors.append(temp / n_repeats)
mean_errors = np.asarray(mean_errors)


anomalies = np.zeros(len(error_list))
print(len(error_list))
anomaly_indices = [42, 43, 44, 45, 51, 74, 102, 206] #743 Malmö Stad
#anomaly_indices = [4, 20, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72] # 39FD
#anomaly_indices = list(range(24, 84))# 40FD
#anomaly_indices = list(range(65, 96))    #454 MS
for i in anomaly_indices:
    anomalies[i] = 1

# calculate CI

means_file = open('means.txt', 'a')
for i in range(n_repeats):
    candidate_values2 = np.linspace(stds[i] / 50, 2 * stds[i], 100)
    f = PlotEvaluationMetrics.find_best_theta(candidate_values2, anomalies, error_list[:, i], 0)
    s = str(f)
    s += '\n'
    means_file.write(s)
means_file.flush()

std = mean_errors.std()
#candidate_values = np.linspace(std / 3, 3 * std, 50)
candidate_values = np.linspace(std / 50, 2 * std, 100)
PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, mean_errors, 1)


""" OLD
plt.plot(range(len(errors)), errors)
plt.show()
anomalies = np.zeros(len(errors))
anomaly_indices = [42, 43, 44, 45, 51, 74, 102, 206] #Malmö Stad 743
#anomaly_indices = range(5, len(errors)) #Fordonsdata 137
for i in anomaly_indices:
    anomalies[i] = 1
   # print(predictions[i])
    #print(x_test[i])
std = errors.std()


candidate_values = np.linspace(std/4 ,  2 * std, 100)
PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, errors)
"""