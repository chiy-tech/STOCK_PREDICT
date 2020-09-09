import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import time

# Using Facebook's Prophet to predict 
from fbprophet import Prophet


def Prophet_model(data):
	new_data=data.loc[:,['Date','Close']]
	new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
	new_data.index = new_data['Date']
	# rename to Prophet's 'y' & 'ds' columns
	new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
	# 75% train set and 25% valid set.
	train = new_data[:1887]
	valid = new_data[1887:]
	Prophet_model = Prophet()
	Prophet_model.fit(train)

	#predictions
	close_prices = Prophet_model.make_future_dataframe(periods=len(valid))
	forecast = Prophet_model.predict(close_prices)

	forecast_valid = forecast['yhat'][1887:]
	valid['Predictions'] = forecast_valid.values

	return train,valid











