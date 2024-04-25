import os
import numpy as np

data_indexes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def concat_data(data_indexes=data_indexes):
    X_train = []
    y_train = []
    for index in data_indexes:
        current_X = np.load(f'preprocessing/X_train{index}.npy')
        current_y = np.load(f'preprocessing/y_train{index}.npy')
        print(index, current_X.shape, current_y.shape)
        X_train.append(current_X)
        y_train.append(current_y)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    np.save('preprocessing/X_train_combined.npy', X_train)
    np.save('preprocessing/y_train_combined.npy', y_train)


if __name__ == '__main__':
    # concat_data()
    x = np.load('preprocessing/X_train_combined.npy')
    y = np.load('preprocessing/y_train_combined.npy')
    print(x.shape, y.shape)

