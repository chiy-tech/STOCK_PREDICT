from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time

def LSTM_model(data):
	# clean data
	new_data=data.loc[:,['Date','Close']]
	new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
	new_data.index = new_data.Date
	new_data.drop(['Date'], axis=1, inplace=True)
	dataset = new_data.values

	# 75% train set and 25% valid set.
	train = dataset[0:1887,:]
	valid = dataset[1887:,:]
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(dataset)
	x_train, y_train = [], []
	for i in range(60,len(train)):
		x_train.append(scaled_data[i-60:i,0])
		y_train.append(scaled_data[i,0])
	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

	# create model
	model = Sequential()
	model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
	model.add(LSTM(units=50))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

	inputs = new_data[len(new_data) - len(valid) - 60:].values
	inputs = inputs.reshape(-1,1)
	inputs  = scaler.transform(inputs)

	X_test = []
	for i in range(60,inputs.shape[0]):
		X_test.append(inputs[i-60:i,0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
	# predict
	closing_price = model.predict(X_test)
	closing_price = scaler.inverse_transform(closing_price)
	train = new_data[:1887]
	valid = new_data[1887:]
	valid['Predictions'] = closing_price
	return train,valid















