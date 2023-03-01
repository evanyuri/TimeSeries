# in the development I'm attempting to add the addtional dimensions (features to the model)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
import tensorflow as tf
from pandas.tseries.offsets import DateOffset

df = pd.read_csv('raw.csv')
df = df.dropna()
# print(df)
label_columns = 'CSUSHPISA/CPIAUCSL'
plot_cols = ['MORTGAGE30US', 'CASTHPI', 'CAUR', 'CSUSHPISA/CPIAUCSL']
# print(plot_cols)
num_features = len(plot_cols)
df = df[plot_cols]
df = df.iloc[::-1]
df = df[0::5]
df = df[200:]


index = np.arange(0, len(df), 1, dtype=int)
df = df.set_index(index)

# exit()
def plot_series(series, format="-", start=0, end=None):
    """Helper function to plot our time series"""
    plt.plot(series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

# Let's save the parameters of our time series in the dataclass
# class G:
OUT_STEPS = 10
SERIES = df
# TIME = time
SPLIT_TIME = 15 # on day 1100 the training period will end. The rest will belong to the validation set
WINDOW_SIZE = 50 # how many data points will we take into account to make our prediction
BATCH_SIZE = 1 # how many items will we supply per batch
SHUFFLE_BUFFER_SIZE = 1 # we need this parameter to define the Tensorflow sample buffer
epochs = 100 
    
# # plot the series
# plt.figure(figsize=(10, 6))
# plot_series(G.TIME, G.SERIES)
# plt.show()

def train_val_split(series, time_step=SPLIT_TIME):
	"""Divide the time series into training and validation set"""
	series_train = series[:time_step]
	series_valid = series[time_step:]

	return series_train, series_valid

def windowed_dataset(series, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER_SIZE):
	"""
	We create time windows to create X and y features.
	For example, if we choose a window of 30, we will create a dataset formed by 30 points as X
	"""
	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	# dataset = dataset.shuffle(shuffle_buffer)
	dataset = dataset.map(lambda window: (window[:-1], window[-1]))
	dataset = dataset.batch(batch_size).prefetch(1)
	return dataset


dataset = windowed_dataset(SERIES)
# print(dataset)


# for element in dataset.as_numpy_iterator(): 
#   print(element) 
# we divide into training and validation set
series_train, series_valid = train_val_split(SERIES)


def create_uncompiled_model():
  # define a sequential model
  model = tf.keras.models.Sequential([ 
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),

    tf.keras.layers.Dense(num_features, activation='relu', kernel_initializer=tf.initializers.zeros(), name=f'layer_1'),  
    tf.keras.layers.Dense(num_features, activation='relu', kernel_initializer=tf.initializers.zeros(), name=f'layer_2'),  
    tf.keras.layers.Reshape([OUT_STEPS, num_features])

    tf.keras.layers.Dense(num_features, name=f'layer_final'),  

]) 

  return model




def create_model():
    tf.random.set_seed(51)
    model = create_uncompiled_model()
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mae"])  
    return model
  
model = create_model()

# we train for 20 epochs with and assign the callback
history = model.fit(dataset, epochs=epochs)


def future_data_formatter(series, window_size=WINDOW_SIZE):
	#"""This function converts the input series into a dataset with time windows for forecasting"""
    ds = tf.data.Dataset.from_tensor_slices(cut_series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(1).prefetch(1)
    return ds


cut_series = SERIES[-WINDOW_SIZE:] 
# print('cut series start', cut_series)
batch = future_data_formatter(cut_series)
# for element in batch.as_numpy_iterator(): 
#   print(element) 
import random

rand = random.randint(0, 200)
a = df.loc[rand:rand + WINDOW_SIZE] 
print(a)


predicted = pd.DataFrame(columns=plot_cols)
n_features =1
forcast_points = 5
for i in range(forcast_points):
  [forecast] = model.predict(batch)
  # print(forecast)
  rand = random.randint(0, 200)
  a = df.loc[rand:rand + WINDOW_SIZE] 
  predicted = predicted.append(pd.DataFrame(forecast, columns=predicted.columns))
  batch = future_data_formatter( a)


ints = len(df)
# print(ints)
# print(len(forecast))
future_indicies = np.arange(ints, ints + len(predicted), 1, dtype=int)
predicted = predicted.set_index(future_indicies)




plt.plot(df[label_columns], label="past")
plt.plot(predicted[label_columns], label="predicted")
plt.show()




