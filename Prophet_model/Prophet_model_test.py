import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time

from Prophet_model import Prophet_model

def main():
	# getting data for Apple
	data = pd.read_csv('../Stock_data/AAPL.csv')
	train,valid = Prophet_model(data)
    # plot Prophet model-APPLE
	plt.figure(figsize=(13,9))
	plt.subplot(2,2,1)
	plt.title('Prophet model-APPLE')
	plt.plot(train['ds'].values,train['y'].values,label='train-data')
	plt.plot(valid['ds'],valid['y'],c = 'r',label='valid-data')
	plt.plot(valid['ds'],valid['Predictions'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close')
	plt.legend()
	#plt.show()
	# getting data for Amazon
	data2 = pd.read_csv('../Stock_data/AMZN.csv')
	train,valid = Prophet_model(data2)
	# plot Prophet model-AMAZON
	plt.subplot(2,2,2)
	plt.title('Prophet model-AMAZON')
	plt.plot(train['ds'].values,train['y'].values,label='train-data')
	plt.plot(valid['ds'],valid['y'],c = 'r',label='valid-data')
	plt.plot(valid['ds'],valid['Predictions'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close')
	plt.legend()
	# getting data for Walmat
	data3 = pd.read_csv('../Stock_data/WMT.csv')
	train,valid = Prophet_model(data3)
	# plot Prophet model-WMT
	plt.subplot(2,2,3)
	plt.title('Prophet model-WMT')
	plt.plot(train['ds'].values,train['y'].values,label='train-data')
	plt.plot(valid['ds'],valid['y'],c = 'r',label='valid-data')
	plt.plot(valid['ds'],valid['Predictions'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close')
	plt.legend()

	# getting data for Google
	data4 = pd.read_csv('../Stock_data/GOOGL.csv')
	plt.subplot(2,2,4)
	train,valid = Prophet_model(data4)
	# plot Prophet model-GOOGLE
	plt.title('Prophet model-GOOGLE')
	plt.plot(train['ds'].values,train['y'].values,label='train-data')
	plt.plot(valid['ds'],valid['y'],c = 'r',label='valid-data')
	plt.plot(valid['ds'],valid['Predictions'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close')
	plt.legend()
	#plt.show()
	# save result
	plt.savefig('Prophet_model_result.png')



if __name__ == '__main__':
    main()