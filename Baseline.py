from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, Dropout
from keras import initializers
from keras import optimizers
from keras.layers import LeakyReLU

def build_baseline(lookback, n_nodes, n_features):
    # Number of units in each layer approximately n_features / 2
    ###################################################
    #This change right here

    #n_units = round(n_features * (2/3)) + 1
    model = Sequential()
    # return_sequences = True makes output 3D array
    model.add(SimpleRNN(5, input_shape=(lookback, n_features),
                        kernel_initializer=initializers.RandomNormal(),
                        return_sequences=False))
    model.add(LeakyReLU(0.3))
    # model.add(SimpleRNN(10, input_shape=(lookback, n_features), return_sequences=False))
    # model.add(Dropout(0.5))
    # model.add(SimpleRNN(10, return_sequences=False))
    model.add(Dropout(0.4))
    # model.add(Dropout(rate=0.01))
    model.add(Dense(1))  # model.add(Dense(1, activation='relu))
    model.add(LeakyReLU(0.3))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam)
    return model