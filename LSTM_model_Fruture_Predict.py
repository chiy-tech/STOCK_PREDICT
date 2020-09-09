from LSTM_model import LSTM_model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time
import os



def main():
	# read data
	data = pd.read_csv('Stock_data/AAPL.csv')
	# clean data
	new_data=data.loc[:,['Date','Close']]
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
	# create the LSTM model
	model = Sequential()
	model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
	model.add(LSTM(units=50))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

	# create a loop to use the LSTM model to predict stock in November

	# n = the number of days you want to predict using LSTM model
	n = 19
	answer = []
	for i in range(0,n):
		if(i == 0):
			inputs = new_data[len(new_data) - 60 + i:].values

		else:
			inputs=new_data[len(new_data) - 60 + i:].values
			inputs=np.append(inputs,answer[0:len(answer)])

		inputs = inputs.reshape(-1,1)
		inputs = scaler.transform(inputs)
		X_test = []
		X_test.append(inputs[0:60,0])
		X_test = np.array(X_test)
		X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
		closing_price = model.predict(X_test)
		closing_price = scaler.inverse_transform(closing_price)
		answer.append(closing_price[0])
	# result is a null dataframe with only Date column
	result = pd.read_csv('Stock_data/predict_sample.csv').drop('1',axis = 1)
	# input the predict column
	result['predict'] = np.double(answer)
	# real_data is the real dataframe in November
	real_data = pd.read_csv('Stock_data/AAPL_test.csv').loc[:,['Date','Close']]
	real_data['Date'] = pd.to_datetime(real_data.Date,format='%Y-%m-%d')
	result['Data'] = pd.to_datetime(result.Data,format='%Y-%m-%d')
	# Plot the Real data VS Predict
	plt.figure(figsize=(16,9))
	plt.title("Real data VS Predict")
	plt.plot(result.Data,result.predict,label = 'predict')
	plt.plot(real_data.Date,real_data.Close,label = 'real_data')
	plt.xlabel('Date')
	plt.ylabel('Close')
	plt.legend()
	# save result
	plt.savefig('Real_data VS Predict.png')

if __name__ == '__main__':
    main()


















