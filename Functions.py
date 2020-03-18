import pandas as pd
import numpy as np

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

def get_indices(metaLibraryKey):
    if metaLibraryKey == 743:
        return [42, 43, 44, 45, 51, 74, 102, 206]
    elif metaLibraryKey == 39:
        return [4, 20, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
    elif metaLibraryKey == 40:
        return list(range(24, 84))
    else:
        return list(range(65, 96))

def ask_user_to_input_metaLibraryKey():
    print("Please select an index between the following metaLibrarykeys:")
    metaLibraryKeys = ["39 Fordonsdata", "40 Fordonsdata", "454 Malmä Stad", "743 Malmö Stad"]
    for i in range(len(metaLibraryKeys)):
        print(str(i + 1) + ". " + metaLibraryKeys[i])
    index = -1
    metaLibraryKey = 0
    while index < 1 or index > len(metaLibraryKeys):
        index = input()
        if int(index) < 1 or int(index) > len(metaLibraryKeys):
            print("Unvalid index, select again an index between 1 and " + str(len(metaLibraryKeys)))
        else:
            metaLibraryKey = int(metaLibraryKeys[int(index) - 1].split()[0])
        index = int(index)
    return metaLibraryKey

def ask_user_to_select_prediction_model():
    print("Please select a model to train between the following two models:")
    print("RNN", "LSTM", sep="\n")
    choice = ""
    while choice.lower() != "RNN" or choice.lower() != "LSTM":
        choice = input()
        if choice.lower() == "RNN".lower() or choice.lower() == "LSTM".lower():
            if choice.lower() == "RNN".lower():
                return True
            else:
                return False
        else:
            print("Unvalid choice, select either RNN or LSTM")

def get_n_train(metaLibraryKey):
    if metaLibraryKey == 743:
        return 504
    elif metaLibraryKey == 39:
        return 250
    elif metaLibraryKey == 40:
        return 228
    else:
        return 224

def get_csv_file_extension(metaLibraryKey):
    if metaLibraryKey == 454 or metaLibraryKey == 743:
        return 'MS'
    else:
        return 'FD'

def calculate_mean(array, n_repeats):
    array = np.asarray(array)
    array = array.transpose()
    mean = list()
    for i in range(len(array)):
        temp = 0
        for j in range(len(array[1])):
            temp += array[i][j]
        mean.append(temp / n_repeats)
    return array, np.asarray(mean)

def get_config_prediction_models(metaLibraryKey, train_baseline):
    if metaLibraryKey == 743:
        if train_baseline == True:
            n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [5, 10, 4, 4, 4]
        else:
            n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [10, 15, 4, 4, 4]
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

    return [n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size]

def get_config_autoencoder_model(metaLibraryKey):
    if metaLibraryKey == 743:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [4, 10, 2, 4, 2]
    elif metaLibraryKey == 39:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [4, 10, 6, 6, 2]
    elif metaLibraryKey == 454:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [4, 15, 2, 2, 4]
    else:
        n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size = [4, 10, 2, 4, 2]

    return [n_nodes, n_epochs, batch_size, n_online_epochs, online_batch_size]