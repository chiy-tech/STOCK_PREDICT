from general_ML_model import liner_model,knn_model,svm_model,RFTree_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time

def main():
	# getting data
	data = pd.read_csv('../Stock_data/AAPL.csv')
	# loading all the models
	knn_train,knn_valid = knn_model(data)
	liner_train,linear_valid = liner_model(data)
	svm_train,svm_valid=svm_model(data)
	RFT_train,RFT_valid=RFTree_model(data)
	# plot Linear regression
	plt.figure(figsize=(13,9))
	plt.subplot(2,2,1)
	plt.title('linear model')
	plt.plot(liner_train['Date'].values, liner_train['Close'],label = 'train-data')
	plt.plot(linear_valid['Date'].values, linear_valid['Close'], c= 'b',label = 'valid-data')
	plt.plot(linear_valid['Date'].values, linear_valid['prediction'].values, '-', linewidth=2,label='predict line')
	plt.xlabel('Date')
	plt.ylabel('Close Index')
	plt.legend()
	# plot Regression based on k-nearest neighbors
	plt.subplot(2,2,2)
	plt.title('KNN-model')
	plt.plot(knn_train['Date'],knn_train['Close'],label='train-data')
	plt.plot(knn_valid['Date'],knn_valid['Close'],c = 'b',label='valid-data')
	plt.plot(knn_valid['Date'],knn_valid['predict'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close Index')	
	plt.legend()
	# plot Epsilon-Support Vector Regression
	plt.subplot(2,2,3)
	plt.title('SVM-model')
	plt.plot(svm_train['Date'],svm_train['Close'],label='train-data')
	plt.plot(svm_valid['Date'],svm_valid['Close'],c = 'b',label='valid-data')
	plt.plot(svm_valid['Date'],svm_valid['predict'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close Index')	
	plt.legend()
	# plot Random forest regressor
	plt.subplot(2,2,4)
	plt.title('Random Forest-model')
	plt.plot(RFT_train['Date'],RFT_train['Close'],label='train-data')
	plt.plot(RFT_valid['Date'],RFT_valid['Close'],c = 'b',label='valid-data')
	plt.plot(RFT_valid['Date'],RFT_valid['predict'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close Index')
	plt.legend()
	#plt.show()
	# save result
	plt.savefig('general_ML_result.png')
	
if __name__ == '__main__':
    main()