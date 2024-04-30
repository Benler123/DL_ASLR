import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import contextlib
import logging
from models.vae import VAE, loss_function  
from torchsummary import summary
from tqdm import tqdm
import configparser
import argparse
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EPOCHS = 2
# BATCH_SIZE = 32
# EXPERIMENT_NAME = 'VAE_experiment'
# LEARNING_RATE = 0.001
# NUM_FRAMES = 60
# NUM_LANDMARKS = 21
# HIDDEN_SIZE = 256
# LATENT_DIM = 50

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# handler = logging.FileHandler(f'logs/{EXPERIMENT_NAME}.log')
# handler.setLevel(logging.INFO)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)

# logger.addHandler(handler)

def load_data():

    X_train = np.load('../scratch/X_train_combined.npy')

    X_train = X_train.reshape(-1, NUM_FRAMES, NUM_LANDMARKS * 2)  
    return X_train


def train_model(model, X_train, optimizer, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for i in tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
            X_batch = X_train[i:i + batch_size]
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(X_batch)
            loss = loss_function(recon_batch, X_batch, mu, log_var)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()
        
        average_loss = overall_loss / len(X_train)
        logger.info(f'Epoch {epoch + 1} Average Loss: {average_loss:.6f}')
        print(f'Epoch {epoch + 1} Average Loss: {average_loss:.6f}')


def evaluate_model(model, X_test, batch_size):
    model.eval()  
    with torch.no_grad():
        test_loss = 0
        for i in tqdm(range(0, len(X_test), batch_size), desc='Testing'):
            X_batch = X_test[i:i + batch_size]
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            recon_batch, mu, log_var = model(X_batch)
            loss = loss_function(recon_batch, X_batch, mu, log_var)
            test_loss += loss.item()
        
        average_loss = test_loss / len(X_test)
        print(f'Average Test Loss: {average_loss:.6f}')
        logger.info(f'Average Test Loss: {average_loss:.6f}')

def pass_in_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.vae', help='Path to config file')
    args = parser.parse_args()
    return args.config

def parse_arguments():
    config = configparser.ConfigParser()
    config.read('config.vae')
    
    model = config['DEFAULT']['MODEL']
    experiment_name = config['DEFAULT']['EXPERIMENT_NAME']
    epochs = int(config['DEFAULT']['EPOCHS'])
    batch_size = int(config['DEFAULT']['BATCH_SIZE'])
    learning_rate = float(config['DEFAULT']['LEARNING_RATE'])
    num_frames = int(config['DEFAULT']['NUM_FRAMES'])
    num_landmarks = int(config['DEFAULT']['NUM_LANDMARKS'])
    hidden_size = int(config['DEFAULT']['HIDDEN_SIZE'])
    latent_dim = int(config['DEFAULT']['LATENT_DIM'])

    return model, experiment_name, epochs, batch_size, learning_rate, num_frames, num_landmarks, hidden_size, latent_dim

if __name__ == '__main__':

    MODEL_NAME, EXPERIMENT_NAME, EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_FRAMES, NUM_LANDMARKS, HIDDEN_SIZE, LATENT_DIM = parse_arguments()
    gc.collect()
    torch.cuda.empty_cache()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    handler = logging.FileHandler(f'logs/{EXPERIMENT_NAME}.log')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    X_train = load_data()
    X_train, X_test = train_test_split(X_train, test_size=0.2, random_state=42)

    vae_model = VAE(NUM_LANDMARKS, NUM_FRAMES, HIDDEN_SIZE, LATENT_DIM).to(device)
    optimizer = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)
    train_model(vae_model, X_train, optimizer, EPOCHS, BATCH_SIZE)

    evaluate_model(vae_model, X_test, BATCH_SIZE)

    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'X_test shape: {X_test.shape}')
