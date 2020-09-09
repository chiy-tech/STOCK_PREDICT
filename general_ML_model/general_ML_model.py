import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# String to Datetime
def strToDatetime(string):
    Datetime = datetime.datetime.strptime(string,'%Y-%m-%d')
    return Datetime
# Datetime to timestamp
def to_timestamp(datetime):
    stamp = time.mktime(time.strptime(datetime.strftime('%Y-%m-%d'), '%Y-%m-%d'))
    return stamp

def slice_data(data):
	# clean data
	new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close','timestamp'])
	new_data=data.loc[:,['Date','Close','timestamp']]
	new_data['Date'] = new_data['Date'].apply(strToDatetime)
	new_data['timestamp'] = new_data['Date'].apply(to_timestamp)
	# 75% train set and 25% valid set.
	train = new_data[:1887]
	valid = new_data[1887:]
	scaler = MinMaxScaler(feature_range=(0, 1))
	x_train = train.drop(['Close','Date'], axis=1)
	y_train = train['Close']
	x_valid = valid.drop(['Close','Date'], axis=1)
	y_valid = valid['Close']
	x_train_scaled = scaler.fit_transform(x_train)
	x_train = pd.DataFrame(x_train_scaled)
	x_valid_scaled = scaler.fit_transform(x_valid)
	x_valid = pd.DataFrame(x_valid_scaled)
	return x_train,x_valid,y_train,y_valid,train,valid

# Linear regression
def liner_model(data):
	new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close','timestamp'])
	new_data=data.loc[:,['Date','Close','timestamp']]
	new_data['Date'] = new_data['Date'].apply(strToDatetime)
	new_data['timestamp'] = new_data['Date'].apply(to_timestamp)
	train = new_data[0:1887]
	valid = pd.DataFrame()
	fit = stats.linregress(train['timestamp'], train['Close'])
	new_data['prediction'] = fit.slope * new_data['timestamp'] + fit.intercept
	
	valid = new_data[1887:]
	return train,valid


# Regression based on k-nearest neighbors
from sklearn.neighbors import KNeighborsRegressor
def knn_model(data):
	x_train,x_valid,y_train,y_valid,train,valid = slice_data(data)
	model = KNeighborsRegressor(5)
	model.fit(x_train, y_train)
	valid['predict'] = model.predict(x_valid)
	return train,valid


#  Epsilon-Support Vector Regression
from sklearn.svm import SVR
def svm_model(data):
	x_train,x_valid,y_train,y_valid,train,valid = slice_data(data)
	model = SVR(kernel='rbf', C=1, gamma='auto')
	model.fit(x_train, y_train)
	valid['predict'] = model.predict(x_valid)
	return train,valid


#  Random forest regressor
from sklearn.ensemble import RandomForestRegressor
def RFTree_model(data):
	x_train,x_valid,y_train,y_valid,train,valid = slice_data(data)

	RFmodel = RandomForestRegressor(30, max_depth=4)
	RFmodel.fit(x_train, y_train)
	#print(RFmodel.score(x_valid, y_valid))
	valid['predict'] = RFmodel.predict(x_valid)
	return train,valid





