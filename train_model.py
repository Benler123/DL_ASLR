import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models.base_nn as base_nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    try:
        X_train = np.load('preprocessing/X_train.npy')
        y_train = np.load('preprocessing/y_train.npy')
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
        print(f'Epoch {epoch} Loss {loss_list[-1]} Accuracy {train_acc_list[-1]}')
    return model, loss_list, train_acc_list

def generate_save_plots(loss, accuracy):
    plt.figure()
    plt.plot(accuracy)
    plt.legend(['train'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('train_acc.png')

    plt.figure()
    plt.plot(loss)
    plt.legend(['train'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('train_loss.png')

if __name__ == '__main__':

    X_train, y_train = load_data()
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')

    NN_model = base_nn.NN_model(len(X_train[0]) * 2, len(y_train[1]))
    NN_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NN_model.parameters(), lr=0.0001)
    model, loss, accuracy = train_model(NN_model, X_train, y_train, criterion, optimizer, 40, 32)
    generate_save_plots(loss, accuracy)



