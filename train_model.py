import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models.base_nn as base_nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import contextlib
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


EPOCHS=1
BATCH_SIZE=32
EXPERIMENT_NAME='nn_baseline'
LEARNING_RATE=0.0001

def load_data():
    try:
        X_train = np.load('preprocessing/X_train_combined.npy')
        y_train = np.load('preprocessing/y_train_combined.npy')
    except:
        print('Data not found. Please run the preprocessing script first.')
        raise Exception('Data not found')
    return X_train, y_train

def train_model(model, X_train, y_train, criterion, optimizer, epochs, batch_size):
    loss_list = []
    train_acc_list = []
    for epoch in range(epochs):
        epoch_loss = []
        epoch_train_acc = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size, :]
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            X_batch = X_batch.view(X_batch.shape[0], -1, X_batch.shape[1] * X_batch.shape[2])
            optimizer.zero_grad()

            output = model(X_batch).squeeze()
            y_pred_train = torch.argmax(output, dim=1)
            y_actual_labels = torch.argmax(y_batch, dim=1)
            train_acc = (y_pred_train == y_actual_labels).float().mean()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_train_acc.append(train_acc)
        loss_list.append(sum(epoch_loss) / len(epoch_loss))
        train_acc_list.append(sum(epoch_train_acc) / len(epoch_train_acc))
        print(f'Epoch {epoch + 1} Loss {loss_list[-1]} Accuracy {train_acc_list[-1]}')
    return model, loss_list, train_acc_list

def test_model(model, X_test, y_test, criterion):
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    X_test = X_test.view(X_test.shape[0], -1, X_test.shape[1] * X_test.shape[2])
    output = model(X_test).squeeze()
    y_pred_test = torch.argmax(output, dim=1)
    y_actual_labels = torch.argmax(y_test, dim=1)
    test_acc = (y_pred_test == y_actual_labels).float().mean()
    loss = criterion(output, y_test)
    print(f'Test Loss {loss.item()} Accuracy {test_acc}')

def generate_save_plots(experiment_name, loss, accuracy):
    plt.figure()
    plt.plot(accuracy)
    plt.title('Training Accuracy')
    plt.legend(['train'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(f'{experiment_name}_train_acc.png')

    plt.figure()
    plt.plot(loss)
    plt.title('Training Loss')
    plt.legend(['train'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'{experiment_name}_train_loss.png')

def summarize_model(model, input_shape, experiment_name=EXPERIMENT_NAME):
    with open(f'{experiment_name}_summary.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            print(summary(model, input_shape))

if __name__ == '__main__':
    X_data, y_data = load_data()
    X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.2)

    print(f'X_data shape: {X_data.shape}')
    print(f'y_data shape: {y_data.shape}')

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    NN_model = base_nn.NN_model(len(X_train[0]) * 2, len(y_train[1]))
    NN_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NN_model.parameters(), lr=LEARNING_RATE)
    model, loss, accuracy = train_model(NN_model, X_train, y_train, criterion, optimizer, EPOCHS, BATCH_SIZE)
    summarize_model(model, (BATCH_SIZE, X_train.shape[1] * X_train.shape[2]))
    test_model(model, X_test, y_test, criterion)
    generate_save_plots(EXPERIMENT_NAME, loss, accuracy)



