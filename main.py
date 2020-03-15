from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from LSTM import build_LSTM
from Baseline import build_baseline
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import PlotEvaluationMetrics
from scipy.stats import sem, t

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)
#print(len(pd.date_range('2018-08-29', ' 2018-10-27')))
# Define data dimensions
# df_features = pd.read_csv('features_OneHot.csv')


df_features = pd.read_csv('Prediction_454_MS.csv')
labels = pd.read_csv('labels_454_MS.csv')
#print(df_features)


# df_features = pd.read_csv('Prediction_743_MS.csv')
# labels = pd.read_csv('labels_743_MS.csv')
# print(df_features)


# df_features = df_features.drop(['ExecutionDaySin', 'ExecutionDayCos', 'ExecutionMonthSin', 'ExecutionMonthCos',
# 'ExecutionWeekDaySin', 'ExecutionWeekDayCos'], axis=1)
# df_features = df_features['InsertSum']
lookback = 30
#n_train = 504 #743MS
#n_train = 250 #39FD
#n_train = 228 #40FD
n_train = 224  #454MS
n_features = 1  # df_features.shape[1]
num_obs = df_features.shape[0] - lookback
n_test = num_obs - n_train



def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def transform_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = 1  # df.shape[1]
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


def walk_forward_validation(x_train, x_test, y_train, y_test, config):
    lookback, n_nodes, n_epochs, n_batch = config
    #model = build_baseline(lookback, n_nodes, n_features)
    model = build_LSTM(lookback, n_nodes, n_features)
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=2, shuffle=False,
                        validation_data=(x_test, y_test))
    predictions_train = list()
    for i in range(0, len(x_train), n_batch):
        x = x_train[i: i + n_batch, :, :]
        y = y_train[i: i + n_batch]
        y_hat_train = model.predict(x, batch_size=n_batch)
        predictions_train.append(y_hat_train)
    predictions_train = np.asarray(predictions_train)
    predictions_train = predictions_train.reshape(num_obs - n_test, 1)
    y_train = y_train.to_numpy()
    error_list_train = list()
    for i in range(len(predictions_train)):
        error_list_train.append(mean_squared_error(predictions_train[i], y_train[i]))
    error_list_train = np.asarray(error_list_train)
    score_train = mean_squared_error(predictions_train, y_train)


    predictions = list()
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.to_numpy()
    y_test_old = 0
    x_test_old = 0
    on_e = 2
    on_e = 4
    on_b = 4
    for i in range(0, len(x_test), on_b):
        x = x_test[i: i + on_b, :, :]
        # x = x_test[i].reshape(1, 30, 42)
        y = y_test[i:i + on_b]
        if i > 0:
            model.fit(x_test_old, y_test_old, epochs=on_e, batch_size=on_b, shuffle=False, verbose=2)
            model.reset_states()
        yhat = model.predict(x, batch_size=on_b)
        predictions.append(yhat)
        x_test_old = x
        y_test_old = y

    predictions = np.asarray(predictions)
    predictions = predictions.reshape(num_obs - n_train, 1)
    err_list = list()
    for i in range(len(predictions)):
        err_list.append(mean_squared_error(predictions[i], y_test[i]))
    err_list = np.asarray(err_list)
    error = mean_squared_error(predictions, y_test)
    print(' > %.3f' % error)
    return predictions_train, error, err_list, predictions



def repeat_evaluate(x_train, x_test, y_train, y_test, config, n_repeats=1):
    # convert config to a key
    key = config
    errors = list()
    scores = list()
    pred_train = list()
    pred = list()
    # fit and evaluate the model n times
    stds = list()
    for i in range(n_repeats):
        predictions_train, score, error, predictions = walk_forward_validation(x_train, x_test, y_train, y_test, config)
        predictions = predictions[0:, 0]
        predictions_train = predictions_train[0:, 0]
        pred.append(predictions)
        scores.append(score)
        errors.append(error)
        stds.append(error.std())
        pred_train.append(predictions_train)
        checkList = np.zeros(len(predictions))
    scores = np.asarray(scores)
    errors = np.asarray(errors)
    result = np.mean(scores)
    pred = np.asarray(pred)
    pred_train = np.asarray(pred_train)
    pred_train = pred_train.transpose()
    mean_pred_train = list()
    pred = pred.transpose()
    for i in range(len(pred_train)):
        temp = 0
        for j in range(len(pred_train[1])):
            temp += pred_train[i][j]
        mean_pred_train.append(temp/n_repeats)
    mean_pred_train = np.asarray(mean_pred_train)
    plt.plot(range(len(y_train)), y_train, 'b')  # plotting t, a separately
    plt.plot(range(len(mean_pred_train)), mean_pred_train, 'r')  # plotting t, b separately
    plt.show()
    mean_predictions = list()
    mean_errors = list()
    errors = errors.transpose()
    for i in range(len(pred)):
        temp = 0
        for j in range(len(pred[1])):
            temp += pred[i][j]
        mean_predictions.append(temp / n_repeats)
    mean_predictions = np.asarray(mean_predictions)
    plt.plot(range(len(y_test)), y_test, 'b')  # plotting t, a separately
    plt.plot(range(len(mean_predictions)), mean_predictions, 'r')  # plotting t, b separately
    plt.show()
    for i in range(len(errors)):
        temp = 0
        for j in range(len(errors[1])):
            #print(errors[i][j])
            temp += errors[i][j]
        mean_errors.append(temp / n_repeats)
    mean_errors = np.asarray(mean_errors)
    #print("shape: ", errors.shape)
    anomalies = np.zeros(len(errors))
    #anomaly_indices = [42, 43, 44, 45, 51, 74, 102, 206] #743 Malmö Stad
    #anomaly_indices = [4, 20, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72] # 39FD
    #anomaly_indices = list(range(24, 84))# 40FD
    anomaly_indices = list(range(65, 96))  # 454 MS
    for i in anomaly_indices:
        anomalies[i] = 1
    std = mean_errors.std()
    candidate_values = np.linspace(std / 50, 2 * std, 100)
    # candidate_values = [0.045, 0.05, 0.055, 0.06, 0.06, 0.065, 0.07]
    # calculate CI
    means_file = open('means.txt', 'a')
    for i in range(n_repeats):
        candidate_values2 = np.linspace(stds[i] / 50, 2 * stds[i], 100)
        f = PlotEvaluationMetrics.find_best_theta(candidate_values2, anomalies, errors[:, i], 0)
        s = str(f)
        s += '\n'
        means_file.write(s)
    means_file.flush()

    PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, mean_errors, 1)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


def grid_search(x_train, x_test, y_train, y_test, cfg_list):
    # evaluate configs
    scores = [repeat_evaluate(x_train, x_test, y_train, y_test, cfg) for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def model_configs(parameters):
    # define scope of configs
    configs = list()
    lookback = parameters[0]
    n_nodes = parameters[1]
    n_epochs = parameters[2]
    n_batch = parameters[3]
    # n_optimizers = parameters[4]
    # create configs
    for i in lookback:
        for j in n_nodes:
            for k in n_epochs:
                for l in n_batch:
                    # for m in n_optimizers:
                    temp = [i, j, k, l]
                    configs.append(temp)
    print('Total configs: %d' % len(configs))
    return configs


"""
def train_evaluate_optimal(config, x_train, x_test, y_train, y_test):
    lookback, n_nodes, n_epochs, n_batch = config

    #model = build_baseline(lookback, n_nodes, n_features)
    model = build_LSTM(lookback, n_nodes, n_features)
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=1, shuffle=False,
                        validation_data=(x_test, y_test))
    predictions = list()
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.to_numpy()
    y_test_old = 0
    x_test_old = 0
    for i in range(0, len(x_test), 4):
        x = x_test[i : i + 4, :, :]

        # x = x_test[i].reshape(1, 30, 42)
        y = y_test[i: i + 4]
        if i > 0:
            model.fit(x_test_old, y_test_old, epochs=4, batch_size=2, shuffle=False, verbose=1)
            model.reset_states()
        yhat = model.predict(x, batch_size=4)
        predictions.append(yhat)
        x_test_old = x
        y_test_old = y

    predictions = np.asarray(predictions)
    predictions = predictions.reshape(num_obs - n_train, 1)
    errors = list()
    for i in range(len(predictions)):
        errors.append(mean_squared_error(predictions[i], y_test[i]))
    errors = np.asarray(errors)
    anomalies = np.zeros(len(errors))
    anomaly_indices = [42, 43, 44, 45, 51, 74, 102, 206]
    # anomaly_indices = [12]
    #anomaly_indices = range(len(errors))

    for i in anomaly_indices:
        anomalies[i] = 1
    #print(anomalies)
    #print(errors[41:52])
    std = errors.std()
    candidate_values = np.linspace(std/2, 2* std, 50)
    #candidate_values = [0.045, 0.05, 0.055, 0.06, 0.06, 0.065, 0.07]
    PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, errors)


    #print(errors)
    #plt.plot(range(len(errors)), errors)
    #plt.show()

    #Draw loss curves
    #Draw predicted vs test
    #plt.plot(range(len(y_test)), y_test, 'b')  # plotting t, a separately
    #plt.plot(range(len(predictions)), predictions, 'r')  # plotting t, b separately
    #plt.show()
"""

arr = np.array([[lookback], [10], [15], [4]])

configs = model_configs(arr)

labels = labels[lookback - 1:]
data = transform_to_supervised(df_features, n_in=lookback)
data = np.array(data)

x_train = data[:n_train, :-n_features]
y_train = labels[:n_train]
x_test = data[n_train:, :-n_features]
y_test = labels[n_train:]
x_train = x_train.reshape((n_train, lookback, n_features))
x_test = x_test.reshape((num_obs - n_train, lookback, n_features))


scores = grid_search(x_train, x_test, y_train, y_test, configs)

print('---------------------------------  done  ---------------------------------------')
print('printing top 5 configurations')

for cfg, error in scores[:5]:
    print(cfg, error)

# Evaluate and visualize optimal config
# train_evaluate_optimal(scores[0][0], x_train, x_test, y_train, y_test)
