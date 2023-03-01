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

df = pd.read_csv('weather.csv')
df = df.dropna()
# print(df)
label_columns = 'MaxTemp'
plot_cols = ["MinTemp", "MaxTemp", "Humidity3pm",  "Pressure9am",]
num_features = len(plot_cols)
df = df[plot_cols]
# df = df[:200]
# series = np.array(df, dtype=np.float32)
# print(series)
# time = np.arange(len(series), dtype="float32")
# print(time)


# tds = tf.expand_dims(df, axis=-1)
# print(tds)

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
WINDOW_SIZE = 100 # how many data points will we take into account to make our prediction
BATCH_SIZE = 1 # how many items will we supply per batch
SHUFFLE_BUFFER_SIZE = 1 # we need this parameter to define the Tensorflow sample buffer
epochs = 100   
forcast_points = 200

    
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


for element in dataset.as_numpy_iterator(): 
  print(element) 
# we divide into training and validation set
series_train, series_valid = train_val_split(SERIES)


def create_uncompiled_model():
  # define a sequential model
  model = tf.keras.models.Sequential([ 
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    # tf.keras.layers.Reshape([OUT_STEPS, num_features])
  ]) 

  return model


class EarlyStopping(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    
    if(logs.get('mae') < 0.03):
      print("\nMAEthreshold reached. Training stopped.")
      self.model.stop_training = True

# Let's create an object of our class and assign it to a variable
early_stopping = EarlyStopping()


def create_model():
    tf.random.set_seed(51)
  
    model = create_uncompiled_model()
  
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mae"])  
    return model
  
model = create_model()

# we train for 20 epochs with and assign the callback
history = model.fit(dataset, epochs=epochs, callbacks=[early_stopping])

def compute_metrics(true_series, forecast):
    #"""Helper to print MSE and MAE"""
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae

def future_data_formatter(series, window_size=WINDOW_SIZE):
	#"""This function converts the input series into a dataset with time windows for forecasting"""
    ds = tf.data.Dataset.from_tensor_slices(cut_series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(1).prefetch(1)
    return ds


cut_series = SERIES[-WINDOW_SIZE:] 
print('cut series start', cut_series)
batch = future_data_formatter(cut_series)
for element in batch.as_numpy_iterator(): 
  print(element) 

predicted = pd.DataFrame(columns=plot_cols)
n_features =1
print('start batch: ', batch)
for i in range(forcast_points):
  forecast = model.predict(batch)
  [[forecast]] = forecast
  predicted.loc[cut_series.index.max()+1] = forecast

  # print(forecast)
  # print('index max: ', cut_series.index.max())
  cut_series.loc[cut_series.index.max()+1] = forecast
  cut_series = cut_series.iloc[1:]
  # print('cut series new', cut_series)

  batch = future_data_formatter(cut_series)
  # print('new batch: ', batch)

print(predicted)

#add filler point for plot
# predicted.iloc[0] = df.loc[df.index.max()]
# print(predicted)
plt.plot(df[label_columns], label="past")
plt.plot(predicted[label_columns], label="predicted")

plt.show()




