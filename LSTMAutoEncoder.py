from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
import pandas as pd
from keras.layers import LeakyReLU
import numpy as np
import Functions
import PlotEvaluationMetrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

metaLibraryKey = Functions.ask_user_to_input_metaLibraryKey()
n_train = Functions.get_n_train(metaLibraryKey)
extension = Functions.get_csv_file_extension(metaLibraryKey)
df_features = pd.read_csv(r'CSV_files\autoencoder_' + str(metaLibraryKey) + extension + '.csv')
lookback = 30
n_features = df_features.shape[1]
num_obs = df_features.shape[0] - lookback
n_test = num_obs - n_train

def build_autoencoder(n_nodes, lookback, n_features):
    if n_nodes == 2:
        r = 0.5
    else:
        r = 0.75
    model = Sequential()
    model.add(LSTM(n_nodes, input_shape=(lookback + 1, n_features)))
    model.add(LeakyReLU(0.3))
    model.add(Dropout(r))
    model.add(RepeatVector(lookback + 1))
    model.add(LSTM(n_nodes, return_sequences=True))
    model.add(LeakyReLU(0.3))
    model.add(Dropout(r))
    model.add(TimeDistributed(Dense(n_features)))
    model.add(LeakyReLU(0.3))
    model.compile(optimizer='adam', loss='mse')
    return model

def calculate_error(predicted, actual):
    errors = list()
    for i in range(0, len(predicted)):
        errors.append(mean_squared_error(predicted[i, :], actual[i, :]))
    return np.asarray(errors)

data = Functions.transform_to_supervised(df_features, n_in=lookback)
data = np.array(data)
x_train = data[:n_train, :]
x_test = data[n_train:, :]
x_train = x_train.reshape((n_train, lookback + 1, n_features))
x_test = x_test.reshape((num_obs - n_train, lookback + 1, n_features))

def repeat_evaluate(config, n_repeats=10):
    n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = config
    error_list = list()
    stds = list()
    for i in range(n_repeats):
        x_test_copy = x_test
        model = build_autoencoder(n_nodes, lookback, n_features)
        model.fit(x_train, x_train, epochs=n_epochs, batch_size=batch_size, verbose=2, shuffle=False)
        predictions = list()
        for i in range(0, len(x_test), online_batch_size):
            x = x_test_copy[i: i + online_batch_size, :, :]
            if i > 0:
                model.fit(x_test_old, x_test_old, epochs=n_online_epochs, batch_size=online_batch_size, shuffle=False,
                          verbose=2)
                model.reset_states()
            yhat = model.predict(x, batch_size=online_batch_size)
            predictions.append(yhat)
            x_test_old = x
        x_test_copy = x_test_copy[:, -1, :]
        predictions = np.asarray(predictions)
        predictions = predictions.reshape(n_test, lookback + 1, n_features)
        predictions = predictions[:, -1, :]
        errors = calculate_error(predictions, x_test_copy)
        error_list.append(errors)
        stds.append(errors.std())
    error_list, mean_errors = Functions.calculate_mean(error_list, n_repeats)
    anomalies = np.zeros(len(error_list))
    anomaly_indices = Functions.get_indices(metaLibraryKey)
    for i in anomaly_indices:
        anomalies[i] = 1
    std_error = mean_errors.std()
    candidate_values = np.linspace(std_error / 50, 2 * std_error, 100)
    PlotEvaluationMetrics.find_best_theta(candidate_values, anomalies, mean_errors)

config = Functions.get_config_autoencoder_model(metaLibraryKey)
repeat_evaluate(config)