import pandas as pd
import glob
import os
import plotly.graph_objects as go
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import yfinance as yf


current_path = os.getcwd()
print(current_path)
files = glob.glob("housingData/*.csv")
print(files)

#initalize DF
df = pd.read_csv(files[0])

for e in files[1:]:
    temp_df = pd.read_csv(e)
    df = df.merge(temp_df, on='DATE', how='outer')
#convert to datetime
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[(df["DATE"]  > "1975-01-01")]
print(df.dtypes)


#get Stock data
sp500 = yf.Ticker("^GSPC").history(period="max")['Close'].to_frame()
sp500 = sp500.reset_index().rename(columns={'Date':'DATE', 'Close':'sp500'})
sp500['DATE'] = sp500['DATE'].dt.date


print(sp500)
sp500['DATE'] = pd.to_datetime(sp500['DATE'])
# plt.plot(sp500['DATE'], sp500['sp500'])
# plt.show()


df = df.merge(sp500, on='DATE', how='inner')
print(df)

# sp500.date
# print(df.dtypes)

# df = df.merge(sp500, left_on='DATE', right_index=True, how='outer')

df = df.sort_values(by='DATE',ascending=False).reset_index(drop=True)
df = df.fillna(method='ffill')


df['CSUSHPISA/CPIAUCSL'] = df['CSUSHPISA']/df["CPIAUCSL"]
df['sp500/CPIAUCSL'] = df['sp500']/df["CPIAUCSL"]

# print(df.describe())

x_cols = ['DATE']
y_cols = list(set(df.columns).difference(set(x_cols)))

for i,y in enumerate(y_cols):
    df[y] = pd.to_numeric(df[y], errors='coerce')

df[y_cols] = minmax_scale(df[y_cols])

fig = go.Figure()
for i,y in enumerate(y_cols):
    fig.add_trace(go.Scatter(x=df[x_cols[0]], y=df[y],mode='lines+markers',name=y))
# df.to_csv('raw.csv')

fig.show()


# import os
# import datetime

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import tensorflow as tf

# date_time = df.pop('DATE')
# timestamp_s = date_time.map(pd.Timestamp.timestamp)

# print(df)
# print(date_time)
# print(timestamp_s)
# plot_features = df[y_cols]

# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1)
# ])