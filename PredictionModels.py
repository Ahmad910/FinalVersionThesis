from sklearn.metrics import mean_squared_error
from Baseline import build_baseline
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from LSTM import build_LSTM
import Functions
import PlotEvaluationMetrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)

# change the metaLibraryKey [39FD, 40FD, 454MS, 743MS].
metaLibraryKey = 40
# set train_baseline to False to train the model with LSTM.
train_baseline = False
n_train = Functions.get_n_train(metaLibraryKey)
extension = Functions.get_csv_file_extension(metaLibraryKey)
print('Prediction_' + str(metaLibraryKey) + extension + '.csv')
df_features = pd.read_csv(r'CSV_files\Prediction_' + str(metaLibraryKey) + extension + '.csv')
labels = pd.read_csv(r'CSV_files\labels_' + str(metaLibraryKey) + extension + '.csv')
lookback = 30
n_features = df_features.shape[1]
num_obs = df_features.shape[0] - lookback
n_test = num_obs - n_train

def calculate_predictions(model, x_set, y_set, n_online_epochs, n_batch):
    predictions = list()
    y_set = y_set.reset_index(drop=True)
    y_set = y_set.to_numpy()
    y_test_old = 0
    x_test_old = 0
    for i in range(0, len(x_set), n_batch):
        x = x_set[i: i + n_batch, :, :]
        y = y_set[i: i + n_batch]
        if i > 0:
            model.fit(x_test_old, y_test_old, epochs=n_online_epochs, batch_size=n_batch, shuffle=False, verbose=2)
            model.reset_states()
        yhat = model.predict(x, batch_size=n_batch)
        predictions.append(yhat)
        x_test_old = x
        y_test_old = y
    predictions = np.asarray(predictions)
    predictions = predictions.reshape(num_obs - n_train, 1)
    error_list = list()
    for i in range(len(predictions)):
        error_list.append(mean_squared_error(predictions[i], y_set[i]))

    error_list = np.asarray(error_list)
    return predictions, error_list



def walk_forward_validation(x_train, x_test, y_train, y_test, config):
    n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = config
    if train_baseline == True:
        model = build_baseline(lookback, n_nodes, n_features)
    else:
        model = build_LSTM(lookback, n_nodes, n_features)

    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2, shuffle=False,
              validation_data=(x_test, y_test))
    predictions_test, error_list_test = calculate_predictions(model, x_test, y_test, n_online_epochs, online_batch_size)
    mean_sq_error_test = mean_squared_error(predictions_test, y_test)
    # print(' > %.3f' % mean_sq_error_test)
    return mean_sq_error_test, error_list_test, predictions_test


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

def repeat_evaluate(x_train, x_test, y_train, y_test, config, n_repeats=1):
    errors_test = list()
    mean_sq_errors_test = list()
    pred_test = list()
    stds_error_test = list()
    for i in range(n_repeats):
        mean_sq_error_test, error_list_test, predictions_test = walk_forward_validation(x_train, x_test, y_train, y_test, config)
        predictions_test = predictions_test[0:, 0]
        pred_test.append(predictions_test)
        mean_sq_errors_test.append(mean_sq_error_test)
        errors_test.append(error_list_test)
        stds_error_test.append(error_list_test.std())
    plot(pred_test, n_repeats, y_test, True)
    errors_test, mean_errors_test = plot(errors_test, n_repeats, y_test)
    anomalies = np.zeros(len(errors_test))
    anomaly_indices = Functions.get_indices(metaLibraryKey)
    for i in anomaly_indices:
        anomalies[i] = 1
    std_error = mean_errors_test.std()
    candidate_values = np.linspace(std_error / 50, 2 * std_error, 100)
    mean_sq_errors_test = np.asarray(mean_sq_errors_test)
    mean_error_test = np.mean(mean_sq_errors_test)
    PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, mean_errors_test)
    # print('> Model[%s] %.3f' % (key, mean_error_test))
    return (config, mean_error_test)

config = Functions.get_config_prediction_models(metaLibraryKey, train_baseline)

labels = labels[lookback - 1:]
data = Functions.transform_to_supervised(df_features, n_in=lookback)
data = np.array(data)
x_train = data[:n_train, :-n_features]
y_train = labels[:n_train]
x_test = data[n_train:, :-n_features]
y_test = labels[n_train:]
x_train = x_train.reshape((n_train, lookback, n_features))
x_test = x_test.reshape((num_obs - n_train, lookback, n_features))
repeat_evaluate(x_train, x_test, y_train, y_test, config)
