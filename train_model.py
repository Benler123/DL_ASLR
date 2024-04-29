import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models.base_nn as base_nn
import models.lstm as lstm
import models.cnn_1d_v1 as cnn_1d_v1
import models.cnn_1d_v2 as cnn_1d_v2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import contextlib
import logging
from torchsummary import summary
import gc




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_FRAMES = 60
NUM_LANDMARKS = 21

EPOCHS=0
BATCH_SIZE=32
MODEL_NAME = "LSTM"
EXPERIMENT_NAME='lstm_double'
LEARNING_RATE=0.001
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_WEIGHT_DECAY = 0.001
LSTM_DROPOUT_PROB = 0.7


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.FileHandler(f'logs/{EXPERIMENT_NAME}.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def load_data():
    try:
        X_train = np.load('../scratch/X_train_combined.npy')
        y_train = np.load('../scratch/y_train_combined.npy')
    except:
        try:
            X_train = np.load('preprocessing/X_train_combined.npy')
            y_train = np.load('preprocessing/y_train_combined.npy')
        except: 
            print('Data not found. Please run the preprocessing script first.')
            raise Exception('Data not found')
    if MODEL_NAME == "CNN" or MODEL_NAME == "LSTM": 
        X_train = X_train.reshape(-1, NUM_FRAMES, NUM_LANDMARKS, 2)
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
            if MODEL_NAME == "NN": 
                X_batch = X_batch.view(X_batch.shape[0], -1, X_batch.shape[1] * X_batch.shape[2])
            optimizer.zero_grad()
            if MODEL_NAME == "LSTM": 
                output, l2_reg = model(X_batch)
                loss = criterion(output.squeeze(), y_batch) + l2_reg
            else: 
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)

            y_pred_train = torch.argmax(output, dim=1)
            y_actual_labels = torch.argmax(y_batch, dim=1)
            train_acc = (y_pred_train == y_actual_labels).float().mean()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_train_acc.append(train_acc)
        loss_list.append(sum(epoch_loss) / len(epoch_loss))
        train_acc_list.append(sum(epoch_train_acc) / len(epoch_train_acc))
        logger.info(f'Epoch {epoch + 1} Loss {loss_list[-1]} Accuracy {train_acc_list[-1]}')
    return model, loss_list, train_acc_list

def test_model(model, X_test, y_test, criterion, batch_size):
    test_loss = []
    test_acc = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            
            if MODEL_NAME == "NN":
                X_batch = X_batch.view(X_batch.shape[0], -1, X_batch.shape[1] * X_batch.shape[2])
            
            if MODEL_NAME == "LSTM":
                output, l2_reg = model(X_batch)
                loss = criterion(output.squeeze(), y_batch) + l2_reg
            else:
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
            
            y_pred_test = torch.argmax(output, dim=1)
            y_actual_labels = torch.argmax(y_batch, dim=1)
            batch_acc = (y_pred_test == y_actual_labels).float().mean()
            
            test_loss.append(loss.item())
            test_acc.append(batch_acc.item())
    
    avg_test_loss = sum(test_loss) / len(test_loss)
    avg_test_acc = sum(test_acc) / len(test_acc)
    
    logger.info(f'Test Loss {avg_test_loss} Accuracy {avg_test_acc}')
    
    return avg_test_acc, avg_test_loss
    
def generate_save_plots(experiment_name, train_loss, test_loss, train_accuracy, test_accuracy):
    train_accuracy = [acc.cpu().numpy() for acc in train_accuracy]
    plt.figure()
    plt.plot(train_accuracy)
    plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test Accuracy')
    plt.title(f'{experiment_name} Training Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(f'graphs/{experiment_name}_train_acc.png')

    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.axhline(y=test_loss, color='r', linestyle='-', label='Test Loss')
    plt.title(f'{experiment_name} Training Loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'graphs/{experiment_name}_train_loss.png')

def summarize_model(model, input_shape, experiment_name=EXPERIMENT_NAME):
    with open(f'{experiment_name}.log', 'a') as f:
        with contextlib.redirect_stdout(f):
            summary(model, input_shape)
        

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f'Model Name: {MODEL_NAME}')
    logger.info(f'Experiment Name: {EXPERIMENT_NAME}')
    logger.info(f'Epochs: {EPOCHS}')
    logger.info(f'Batch Size: {BATCH_SIZE}')
    logger.info(f'Learning Rate: {LEARNING_RATE}')

    if MODEL_NAME == "LSTM":
        logger.info(f'LSTM Hidden Size: {LSTM_HIDDEN_SIZE}')
        logger.info(f'LSTM Number of Layers: {LSTM_NUM_LAYERS}')
        logger.info(f'LSTM Weight Decay: {LSTM_WEIGHT_DECAY}')
        logger.info(f'LSTM Dropout Probability: {LSTM_DROPOUT_PROB}')
    X_train, y_train = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'y_train shape: {y_train.shape}')
    logger.info(f'X_test shape: {X_test.shape}')
    logger.info(f'y_test shape: {y_test.shape}')

    NN_model = base_nn.NN_model(X_train.shape[1] * X_train.shape[2], len(y_train[1])).to(device)

    LSTM_model = lstm.LSTM_model(num_landmarks=NUM_LANDMARKS, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, weight_decay=LSTM_WEIGHT_DECAY, dropout_prob=LSTM_DROPOUT_PROB, output_classes=len(y_train[0])).to(device)

    CNN_model = cnn_1d_v2.CNN1D_model(NUM_LANDMARKS, NUM_FRAMES, len(y_train[0])).to(device)

    current_model = None
    if MODEL_NAME == "CNN": 
        current_model = CNN_model
    if MODEL_NAME == "NN": 
        current_model = NN_model
    if MODEL_NAME == "LSTM": 
        current_model = LSTM_model
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    trained_model, train_loss, train_accuracy = train_model(current_model, X_train, y_train, criterion, optimizer, EPOCHS, BATCH_SIZE)
    test_acc, test_loss = test_model(trained_model, X_test, y_test, criterion, BATCH_SIZE)
    generate_save_plots(EXPERIMENT_NAME, train_loss=train_loss, train_accuracy=train_accuracy, test_accuracy=test_acc, test_loss=test_loss)
    if MODEL_NAME == "NN": 
        summarize_model(trained_model, (BATCH_SIZE, X_train.shape[1] * X_train.shape[2]))
    if MODEL_NAME == "CNN": 
        summarize_model(trained_model, (NUM_FRAMES, NUM_LANDMARKS * 2))
    if MODEL_NAME == "LSTM": 
        summarize_model(trained_model,  (BATCH_SIZE, NUM_FRAMES, NUM_LANDMARKS, 2))

