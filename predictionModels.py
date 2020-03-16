from sklearn.metrics import mean_squared_error
from Baseline import build_baseline
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import PlotEvaluationMetrics
from LSTM import build_LSTM

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)

metaLibraryKey = 743
df_features = pd.read_csv('Prediction_' + str(metaLibraryKey) + '_MS.csv')
labels = pd.read_csv('labels_' + str(metaLibraryKey) + '_MS.csv')
if metaLibraryKey == 743:
    n_train = 504
elif metaLibraryKey == 39:
    n_train = 250
elif metaLibraryKey == 40:
    n_train = 228
else:
    n_train = 224

lookback = 30
n_features = 1
num_obs = df_features.shape[0] - lookback
n_test = num_obs - n_train


def transform_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = 1
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


def calculate_predictions(model, x_set, y_set, n_online_epochs, n_batch, predictions_test):
    predictions = list()
    y_set = y_set.reset_index(drop=True)
    y_set = y_set.to_numpy()
    y_test_old = 0
    x_test_old = 0
    for i in range(0, len(x_set), n_batch):
        x = x_set[i: i + n_batch, :, :]
        y = y_set[i: i + n_batch]
        if predictions_test == True and i > 0:
            model.fit(x_test_old, y_test_old, epochs=n_online_epochs, batch_size=n_batch, shuffle=False, verbose=2)
            model.reset_states()
        yhat = model.predict(x, batch_size=n_batch)
        predictions.append(yhat)
        x_test_old = x
        y_test_old = y
    predictions = np.asarray(predictions)
    if predictions_test == True:
        predictions = predictions.reshape(num_obs - n_train, 1)
    else:
        predictions = predictions.reshape(num_obs - n_test, 1)

    error_list = list()
    for i in range(len(predictions)):
        error_list.append(mean_squared_error(predictions[i], y_set[i]))

    error_list = np.asarray(error_list)
    return predictions, error_list


train_baseline = True


def walk_forward_validation(x_train, x_test, y_train, y_test, config):
    n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = config
    if train_baseline == True:
        model = build_baseline(lookback, n_nodes, n_features)
    else:
        model = build_LSTM(lookback, n_nodes, n_features)

    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2, shuffle=False,
              validation_data=(x_test, y_test))

    predictions_train, error_list_train = calculate_predictions(model, x_train, y_train, n_epochs, online_batch_size,
                                                                False)
    predictions_test, error_list_test = calculate_predictions(model, x_test, y_test, n_online_epochs, online_batch_size,
                                                              True)
    mean_sq_error_test = mean_squared_error(predictions_test, y_test)
    # print(' > %.3f' % mean_sq_error_test)
    return predictions_train, mean_sq_error_test, error_list_test, predictions_test


def plot(predictions, n_repeats, set, show=False):
    predictions = np.asarray(predictions)
    predictions = predictions.transpose()
    mean_predictions = list()
    for i in range(len(predictions)):
        temp = 0
        for j in range(len(predictions[1])):
            temp += predictions[i][j]
        mean_predictions.append(temp / n_repeats)
    mean_predictions = np.asarray(mean_predictions)

    if show == True:
        plt.plot(range(len(set)), set, 'b')  # plotting t, a separately
        plt.plot(range(len(mean_predictions)), mean_predictions, 'r')  # plotting t, b separately
        plt.show()
    return predictions, mean_predictions


def get_indices():
    if metaLibraryKey == 743:
        return [42, 43, 44, 45, 51, 74, 102, 206]
    elif metaLibraryKey == 39:
        return [4, 20, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
    elif metaLibraryKey == 40:
        return list(range(24, 84))
    else:
        return list(range(65, 96))


def repeat_evaluate(x_train, x_test, y_train, y_test, config, n_repeats=2):
    key = config
    errors_test = list()
    mean_sq_errors_test = list()
    pred_train = list()
    pred_test = list()
    stds_error_test = list()
    for i in range(n_repeats):
        predictions_train, mean_sq_error_test, error_list_test, predictions_test = walk_forward_validation(x_train,
                                                                                                           x_test,
                                                                                                           y_train,
                                                                                                           y_test,
                                                                                                           config)
        predictions_train = predictions_train[0:, 0]
        pred_train.append(predictions_train)
        predictions_test = predictions_test[0:, 0]
        pred_test.append(predictions_test)
        mean_sq_errors_test.append(mean_sq_error_test)
        errors_test.append(error_list_test)
        stds_error_test.append(error_list_test.std())
    plot(pred_train, n_repeats, y_train)
    plot(pred_test, n_repeats, y_test, True)
    errors_test, mean_errors_test = plot(errors_test, n_repeats, y_test)
    anomalies = np.zeros(len(errors_test))
    anomaly_indices = get_indices()
    for i in anomaly_indices:
        anomalies[i] = 1
    std_error = mean_errors_test.std()
    candidate_values = np.linspace(std_error / 50, 2 * std_error, 100)
    mean_sq_errors_test = np.asarray(mean_sq_errors_test)
    mean_error_test = np.mean(mean_sq_errors_test)
    PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, mean_errors_test, 1)
    # print('> Model[%s] %.3f' % (key, mean_error_test))
    return (key, mean_error_test)


if metaLibraryKey == 743:
    if train_baseline == True:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [5, 10, 4, 4, 4]
    else:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [10, 15, 4, 4, 4, ]
elif metaLibraryKey == 39:
    if train_baseline == True:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [10, 10, 4, 2, 4]
    else:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [5, 10, 2, 2, 2]
elif metaLibraryKey == 454:
    n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [5, 10, 2, 2, 4]
else:
    if train_baseline == True:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [5, 10, 2, 2, 4]
    else:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [5, 10, 4, 2, 4]

config = [n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size]

labels = labels[lookback - 1:]
data = transform_to_supervised(df_features, n_in=lookback)
data = np.array(data)
x_train = data[:n_train, :-n_features]
y_train = labels[:n_train]
x_test = data[n_train:, :-n_features]
y_test = labels[n_train:]
x_train = x_train.reshape((n_train, lookback, n_features))
x_test = x_test.reshape((num_obs - n_train, lookback, n_features))
repeat_evaluate(x_train, x_test, y_train, y_test, config, n_repeats=2)
print('---------------------------------  done  ---------------------------------------')
