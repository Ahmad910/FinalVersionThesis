from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.layers import LSTM, Dropout
from keras import optimizers


def build_LSTM(lookback, n_nodes, n_features):
    model = Sequential()
    model.add(LSTM(n_nodes, input_shape=(lookback, n_features), return_sequences=True))# return_sequences=True
    model.add(LeakyReLU(0.3))
    if n_nodes == 5:
        r = 0.4
    else:
        r = 0.3
    model.add(Dropout(rate=r))
    model.add(LSTM(n_nodes, return_sequences=False )) #return_sequences=True
    model.add(LeakyReLU(0.3))
    model.add(Dropout(r))
    model.add(Dense(1))
    model.add(LeakyReLU(0.3))
    adam = optimizers.adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam)
    return model