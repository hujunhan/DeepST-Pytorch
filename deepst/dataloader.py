import h5py
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
globalpath=os.path.abspath('..')
print('This project\'s absolute path: ',globalpath)
def del_incomp_data(data_Path):
	reader = h5py.File(data_Path, 'r')
	data = reader['data'].value
	date = reader['date'].value
	reader.close()
	data = np.array(data)
	date = np.array(date)

	date_complete = []  ##完整的天数
	data_complete = []
	date_temp = []  ##用于统计一天是否有48个时间戳
	data_temp = []
	date_str = date[0][0:8]
	for i in range(len(date)):
		if date[i][0:8] == date_str:
			date_temp.append(date[i])
			data_temp.append(data[i])
		else:
			if len(date_temp) == 48:
				date_complete.extend(date_temp)
				data_complete.extend(data_temp)
			date_temp.clear()
			data_temp.clear()
			date_str = date[i][0:8]
			date_temp.append(date[i])
			data_temp.append(data[i])
	# a=input("hello?Check the memory!")
	data_complete = np.array(data_complete)
	print(len(date_complete), len(data_complete))
	# a=input("hello?Check the memory!")
	return data_complete, date_complete


def minmax(data):
	data = 1. * (data - data.min()) / (data.max() - data.min())
	data = data * 2.0 - 1.0
	return data


def get_all_data():
	close_data = []
	period_data = []
	trend_data = []
	Y = []
	timeslots = []
	for year in range(13, 16):
		fname = 'BJ{}_M32x32_T30_InOut.h5'.format(year)
		fname=os.path.join(globalpath,'data\TaxiBJ',fname)
		#print(fname)
		data_temp, date_temp = del_incomp_data(fname)
		close_temp, period_temp, trend_temp, Y_temp, timeslots_temp = get_train_data(data_temp, date_temp)
		close_data.extend(close_temp)
		period_data.extend(period_temp)
		trend_data.extend(trend_temp)
		Y.extend(Y_temp)
		timeslots.extend(timeslots_temp)
	# a=input("hello?Check the memory!")
	close_data = np.array(close_data)
	period_data = np.array(period_data)
	trend_data = np.array(trend_data)
	Y = np.array(Y)

	return close_data, period_data, trend_data, Y, timeslots


# return np.array(close_data),np.array(period_data),np.array(trend_data),np.array(Y),timeslots


def get_train_data(data, date, close=3, period=3, trend=3):
	close_data = []
	period_data = []
	trend_data = []
	Y = data[1008:]
	time_stamps = []
	close = [i for i in range(1, close + 1)]
	period = [48 * j for j in range(1, period + 1)]
	trend = [7 * 48 * j for j in range(1, trend + 1)]
	close_temp = np.array([])
	period_temp = np.array([])
	trend_temp = np.array([])
	data = np.array(data)
	for i in range(1008, (len(data))):
		close_temp = np.vstack([data[i - j] for j in close])
		period_temp = np.vstack([data[i - j] for j in period])
		trend_temp = np.vstack([data[i - j] for j in trend])
		close_data.append(close_temp)
		period_data.append(period_temp)
		trend_data.append(trend_temp)
		time_stamps.append(date[i])
	print(close_data[0].shape)
	print(len(close_data), len(period_data), len(trend_data), len(Y), len(date))
	return close_data, period_data, trend_data, Y, time_stamps


def get_feature_data(date):
	feature_Path=os.path.join(globalpath,'data\TaxiBJ', 'BJ_Meteorology.h5')
	reader = h5py.File(feature_Path, 'r')
	for key in reader.keys():
		print(key, reader[key].shape, reader[key].dtype)
	temperature = reader['Temperature'].value
	weather = reader['Weather'].value
	windspeed = reader['WindSpeed'].value
	feature_date = reader['date'].value
	ws = []
	wr = []
	te = []
	M = dict()
	for i, slot in enumerate(feature_date):
		M[slot] = i
	for i in date:
		feature_index = M[i] - 1
		ws.append([windspeed[feature_index]])
		wr.append(weather[feature_index])
		te.append([temperature[feature_index]])
	ws = np.array(ws)
	wr = np.array(wr)
	te = np.array(te)
	ws = 1. * (ws - ws.min()) / (ws.max() - ws.min())
	te = 1. * (te - te.min()) / (te.max() - te.min())
	merge_data = np.hstack([wr, ws, te])
	return merge_data


if __name__ == '__main__':
	get_all_data()
