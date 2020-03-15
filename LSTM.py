from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.layers import LSTM, Dropout, Bidirectional
from keras import optimizers


def build_LSTM(lookback, n_nodes, n_features):
    #Number of units in each layer is approximately dubble the amout before dropout
    #l1_units = round(n_features * 2/3) + 1
    #l2_units = round((n_features/3)) + 1
    #l3_units = round(n_features/4)
    l1_units = n_nodes
    l2_units = n_nodes

    model = Sequential()
    # return_sequences = True makes output 3D array
    model.add(LSTM(l1_units, input_shape=(lookback, n_features), return_sequences=True))# return_sequences=True
    model.add(LeakyReLU(0.3))
    model.add(Dropout(rate=0.4))
    model.add(LSTM(l2_units, return_sequences=False )) #return_sequences=True
    model.add(LeakyReLU(0.3))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(LeakyReLU(0.3))
    adam = optimizers.adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam)
    return model