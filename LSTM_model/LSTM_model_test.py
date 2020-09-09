import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time

from LSTM_model import LSTM_model
from keras.models import load_model


def main():
	# read data
	data = pd.read_csv('../Stock_data/AAPL.csv')
	train,valid = LSTM_model(data)
	# plot the result
	plt.figure(figsize=(16,9))
	plt.title('LSTM-model')
	plt.plot(train.index,train['Close'],label='train-data')
	plt.plot(valid.index,valid['Close'],c = 'b',label='valid-data')
	plt.plot(valid.index,valid['Predictions'],label='predict-data')
	plt.xlabel('Date')
	plt.ylabel('Close')
	plt.legend()
	# save result
	plt.savefig('LSTM_model_result.png')

if __name__ == '__main__':
    main()


