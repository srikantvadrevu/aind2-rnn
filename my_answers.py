import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from keras.layers import Activation
import string

def rolling_window_output(a):
    return [[row[-1]] for row  in a]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_transform_series(series, window_size):
    rolling_windows = rolling_window(series, window_size)
    X = rolling_windows[:-1]
    y = rolling_window_output(rolling_windows)[1:]
    # reshaping 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1, activation='linear'))
    return model

def cleaned_text(text):
    unique_chars = list(set([i for i in text]))
    all_chars = string.ascii_lowercase
    all_punctuation = [' ', '!', ',', '.', ':', ';', '?']
    non_english_chars = list(set([i for i in unique_chars if i not in all_chars and i not in all_punctuation]))
    # removing non english characters
    for i in non_english_chars:
        text = text.replace(i,' ')        
    return text

def window_transform_text(text, window_size, step_size):
    inputs = []
    outputs = []
    for n in range(0,(len(text)-window_size),step_size):
        inputs.append(text[n:(n+window_size)]) 
        outputs.append(text[n+window_size])
    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model
