from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, Dropout
from keras import initializers
from keras import optimizers
from keras.layers import LeakyReLU

def build_baseline(lookback, n_nodes, n_features):
    model = Sequential()
    model.add(SimpleRNN(n_nodes, input_shape=(lookback, n_features),
                        kernel_initializer=initializers.RandomNormal(),
                        return_sequences=False))
    model.add(LeakyReLU(0.3))
    if n_nodes == 5:
        model.add(Dropout(0.4))
    else:
        model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(LeakyReLU(0.3))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam)
    return model