
from glob import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder

NUM_FRAMES=60
SEQUENCE_NUM = 3

'''
Methods to either upsample or downsample frames to return NUM_FRAMES freams
'''

def interpolate_frames(pq_df, num_frames=NUM_FRAMES):
    '''
    Upsample frames to return NUM_FRAMES frames. Do this by sampling frames at even intervels and copying them

    Parameters:
    pq_df: pandas dataframe containing landmarks
    num_frames: number of frames to return

    Returns:
    new_df: pandas dataframe with NUM_FRAMES frames
    '''
    current_frames = pq_df['frame'].unique()
    start_frame = current_frames[0] 
    needed_frames = num_frames - len(current_frames)
    frame_interval = len(current_frames) // (needed_frames + 1)
    for i in range(1, needed_frames + 1):
        frame = start_frame + i * frame_interval % len(pq_df['frame'].unique())
        end_index_of_first_half = len(pq_df[pq_df['frame'] <= frame])
        pq_df = pd.concat([pq_df[pq_df['frame'] <= frame], pq_df[pq_df['frame'] >= frame]], )
        pq_df = pq_df.reset_index(drop=True)
        pq_df.loc[pq_df.index >= end_index_of_first_half, 'frame'] = pq_df.loc[pq_df.index >= end_index_of_first_half, 'frame'] + 1
    for index, frame in enumerate(pq_df['frame'].unique()):
        pq_df.loc[pq_df['frame'] == frame, 'frame'] = index
    return pq_df

def extract_frames(pq_df, method='uniform' , num_frames=NUM_FRAMES):  
    '''
    Method to handle files with more than num_frames frames. Three different methods to sample num_frames frames:
    1. uniform: sample frames uniformly from the entire video   
    2. end: sample the last num_frames frames
    3. start: sample the first num_frames frames

    I think end my work best

    Parameters:
    pq_df: pandas dataframe containing landmarks
    method: method to sample frames
    num_frames: number of frames to return

    Returns:
    new_df: pandas dataframe with num_frames frames
    '''
    if method == 'uniform':
        total_frames = len(pq_df['frame'].unique())
        first_frame = pq_df['frame'].iloc[0]
        step_size = total_frames // num_frames
        frame_indices_range = range(first_frame, total_frames+first_frame, step_size)
    elif method == 'end':
        unique_frames = pq_df['frame'].unique()
        frame_indices_range = unique_frames[-num_frames:]
    elif method == 'start':
        unique_frames = pq_df['frame'].unique()
        frame_indices_range = unique_frames[:num_frames]
    frame_indices = list(frame_indices_range)[-num_frames:]
    new_df = pq_df[pq_df['frame'].isin(frame_indices)].copy()
    for index, frame in enumerate(frame_indices):
        new_df.loc[new_df['frame'] == frame, 'frame'] = index
    return new_df

def onehot_encode(data, num_classes):
    '''
    one hot encode the labels

    Parameters:
    data: list of labels
    num_classes: number of classes in the dataset

    Returns:
    one hot encoded labels
    '''
    num_labels = num_classes
    one_hot = np.zeros((len(data), num_labels), dtype=int)
    one_hot[np.arange(len(data)), data] = 1
    return one_hot

def labels_mapping(csv_file, key_column, value_column):
    '''
    retrieve labels mapping from csv file. Create a labels dictionary mapping each key to a value 

    Parameters:
    csv_file: path to csv file
    key_column: column containing the keys
    value_column: column containing the values

    Returns:
    dictionary mapping keys to values
    '''
    df = pd.read_csv(csv_file)
    df[key_column] = df[key_column].astype(str)
    dictionary = df.set_index(key_column)[value_column].to_dict()
    return dictionary

def sample_parquets(data_file, n=10000):
    '''
    I added this method to sample the data for training. This is because the data is too large to fit in memory.
    To use all data, set n to the length of the data_file.

    Parameters:
    data_file: list of paths to parquet files
    n: number of samples to return

    Returns:
    data: list of sampled data dfs 
    '''
    indices = np.random.choice(len(data_file), n)
    labels_mapping_dict = labels_mapping('kaggle/input/asl-signs/train.csv', 'sequence_id', 'sign')
    data = [data_file[i] for i in indices]
    labels = [labels_mapping_dict[d.split('/')[-1].split('.')[0]] for d in data]
    label_indexes = convert_labels_to_indexes(labels)
    one_hot_labels = onehot_encode(label_indexes, 250)
    return data, one_hot_labels

def sequential_sample_parquets(data_file, start, end):
    labels_mapping_dict = labels_mapping('kaggle/input/asl-signs/train.csv', 'sequence_id', 'sign')
    indices = range(start, min(len(data_file), end))
    data = [data_file[i] for i in indices]
    labels = [labels_mapping_dict[d.split('/')[-1].split('.')[0]] for d in data]
    label_indexes = convert_labels_to_indexes(labels)
    one_hot_labels = onehot_encode(label_indexes, 250)
    return data, one_hot_labels

def convert_labels_to_indexes(labels):
    '''
    map each sign to an index. This is useful for training the model

    Parameters:
    labels: list of labels

    Returns:
    list of indexes corresponding to the labels
    '''
    json.load(open('kaggle/input/asl-signs/sign_to_prediction_index_map.json'))
    return [json.load(open('kaggle/input/asl-signs/sign_to_prediction_index_map.json'))[label] for label in labels]

def process_data(data_files):
    '''
    ensure that all entries have NUMBER_FRAMES frames

    Parameters:
    data_files: list of data file strings to process

    Returns:
    df_lists: list of dataframes with NUMBER_FRAMES frames
    '''
    df_lists = []
    for i, data  in enumerate(data_files):
        data_df = pd.read_parquet(data)
        data_df = data_df[data_df['type'].isin(['left_hand', 'right_hand'])]
        new_df = None
        if(i % 100 == 0):
            print(i)
        if len(data_df['frame'].unique()) < NUM_FRAMES:
            new_df = interpolate_frames(data_df)
        elif len(data_df['frame'].unique() > NUM_FRAMES):
            new_df = extract_frames(data_df, method='end')
        df_lists.append(new_df)
    return df_lists

def mask_nan_hand(df_list):
    '''
    This dataset consists of one handed signs. Mask the hand that has more nan values

    Parameters:
    df_list: list of dataframes

    Returns:
    new_df: list of dataframes with the hand with more nan values removed
    '''
    new_df = []
    for df in df_list:
        left_hand_landmarks = df[df['type'] == 'left_hand'] 
        right_hand_landmarks = df[df['type'] == 'right_hand']
        count_left_hand_nan = left_hand_landmarks.isna().sum().sum()
        count_right_hand_nan = right_hand_landmarks.isna().sum().sum()
        if count_left_hand_nan > count_right_hand_nan:
            df = df[df['type'] == 'right_hand']
        else:
            df = df[df['type'] == 'left_hand']
        new_df.append(df)
        
    return new_df

def replace_nan(df, val=0.0000001):
    '''
    Replace any nan values with val

    Parameters:
    df: dataframe
    val: value to replace nan values with
    
    Returns:
    numpy array of the dataframe with nan values replaced with val
    '''
    return np.asarray(df[['x', 'y']].fillna(val).values)

def normalize_coords(coords):
    '''set the mean of the data to 0 and std dev to 1'''
    mean = np.mean(coords, axis=0)
    std_dev = np.std(coords, axis=0)
    return (coords - mean) / std_dev


def preprocess_data():
    print('Preprocessing data...')
    data_files = glob("kaggle/input/asl-signs/train_landmark_files/*/*.parquet", recursive=True)
    X_train, y_train = sequential_sample_parquets(data_files, 10000 * SEQUENCE_NUM, 10000 *  (SEQUENCE_NUM + 1))
    X_train = process_data(X_train)
    X_train = mask_nan_hand(X_train)
    X_train = [normalize_coords(replace_nan(x)) for x in X_train]
    mask = [x.shape == X_train[0].shape for x in X_train]
    X_train = [x for m, x in zip(mask, X_train) if m]
    y_train = [y for m, y in zip(mask, y_train) if m]
    for i, _ in enumerate(mask):
        if not i:
            print(_)
    X_train = np.array(X_train)
    np.save(f'preprocessing/X_train{SEQUENCE_NUM}.npy', X_train)
    np.save(f'preprocessing/y_train{SEQUENCE_NUM}.npy', y_train)
    
            

if __name__ == '__main__':
    preprocess_data()

