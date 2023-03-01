
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
label_columns = 'MaxTemp'
plot_cols = ["MinTemp", "MaxTemp", "Humidity3pm",  "Pressure9am",]
df = df[label_columns]
df = df[:200]
series = np.array(df, dtype=np.float32)
print(series)
time = np.arange(len(series), dtype="float32")
# print(time)





def plot_series(time, series, format="-", start=0, end=None):
    """Helper function to plot our time series"""
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

# Let's save the parameters of our time series in the dataclass
class G:
    SERIES = series
    TIME = time
    SPLIT_TIME = 15 # on day 1100 the training period will end. The rest will belong to the validation set
    WINDOW_SIZE = 10 # how many data points will we take into account to make our prediction
    BATCH_SIZE = 4 # how many items will we supply per batch
    SHUFFLE_BUFFER_SIZE = 1 # we need this parameter to define the Tensorflow sample buffer
    
    
# # plot the series
# plt.figure(figsize=(10, 6))
# plot_series(G.TIME, G.SERIES)
# plt.show()

def train_val_split(time, series, time_step=G.SPLIT_TIME):
	"""Divide the time series into training and validation set"""
	time_train = time[:time_step]
	series_train = series[:time_step]
	time_valid = time[time_step:]
	series_valid = series[time_step:]

	return time_train, series_train, time_valid, series_valid

def windowed_dataset(series, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
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


dataset = windowed_dataset(series)
print(dataset)

for element in dataset.as_numpy_iterator(): 
  print(element) 
# we divide into training and validation set
time_train, series_train, time_valid, series_valid = train_val_split(G.TIME, G.SERIES)

def create_uncompiled_model():
  # define a sequential model
  model = tf.keras.models.Sequential([ 
      tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                    input_shape=[None]),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(1),
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
history = model.fit(dataset, epochs=100, callbacks=[early_stopping])

# plot MAE and loss
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['mae'], label='mae')
# plt.plot(history.history['loss'], label='loss')
# plt.legend()
# plt.show()

def compute_metrics(true_series, forecast):
    #"""Helper to print MSE and MAE"""
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae

def model_forecast(model, series, window_size):
	#"""This function converts the input series into a dataset with time windows for forecasting"""
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(1).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# Prediction on the whole series
all_forecast = model_forecast(model, G.SERIES, G.WINDOW_SIZE).squeeze()

# Validation portion
val_forecast = all_forecast[G.SPLIT_TIME - G.WINDOW_SIZE:-1]

# Plot
# plt.figure(figsize=(10, 6))
# plt.plot(series_valid, label="validation set")
# plt.plot(val_forecast, label="predicted")
# plt.xlabel("Timestep")
# plt.ylabel("Value")
# plt.legend()
# plt.show()


# predict one data point into the future
# new_forecast = []

# new_forecast_series = G.SERIES[-G.WINDOW_SIZE:] 

# pred = model.predict(new_forecast_series[np.newaxis])

# plt.figure(figsize=(15, 6))
# plt.plot(G.TIME[-100:], G.SERIES[-100:], label="last 100 points of time series")
# plt.scatter(max(G.TIME)+1, pred, color="red", marker="x", s=70, label="prediction")
# plt.legend()
# plt.show()

#predict multiple datapoints intho the future
pred_list = []
n_features =1

new_forecast_series = G.SERIES[-G.WINDOW_SIZE:] 
# print(new_forecast_series[np.newaxis])
batch = new_forecast_series
print(type(batch))
# print(batch)
for i in range(G.WINDOW_SIZE):   
    pred = model.predict(batch[np.newaxis])
    pred_list.append(pred[:,0][0]) 
    print('pred list: ',pred_list)
    print(type(pred))
    print(pred[:,0])
    # print(batch)
    print(batch)
    batch = np.append(batch,pred[:,0])
    batch = batch[-G.WINDOW_SIZE:]
    print(batch)


print(G.SERIES)
ints = len(G.SERIES)
future_indicies = np.arange(ints, ints + G.WINDOW_SIZE, 1, dtype=int)
print(future_indicies)

# scaler = MinMaxScaler()

plt.plot(G.TIME[-100:], G.SERIES[-100:], label="last 100 points of time series")
pred_list_for_pretty_plots = [G.SERIES[-1]] + pred_list
print(pred_list_for_pretty_plots)
future_indicies_for_pretty_plots = np.append([ints - 1],future_indicies)
print(future_indicies_for_pretty_plots)
plt.plot(future_indicies_for_pretty_plots, pred_list_for_pretty_plots, label="predictions")
plt.show()



