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


df = pd.read_csv('weather.csv')
df = df.dropna()
# print(df)
label = 'MaxTemp'
plot_cols = ["MinTemp", "MaxTemp",  "Rainfall",  "Evaporation",  "Sunshine", "WindSpeed3pm",  "Humidity9am",  "Humidity3pm",  "Pressure9am",  "Pressure3pm",  "Cloud9am",]
df = df[plot_cols]
df = df[label]
#train test split

n = len(df)
train = 0.9
train_split = int(n*train)

train, test = df[0:train_split], df[train_split:]

print(len(df))
print(len(train))
print(len(test))