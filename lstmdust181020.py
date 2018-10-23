# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:47:05 2018

@author: sungj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:33:00 2018

@author: sungj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:26:59 2018

@author: sungj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:47:47 2018

@author: sungj
"""
    
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from keras.layers import TimeDistributed
from keras.layers import LSTM
 
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX = []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)        
	return numpy.array(dataX)
 
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('C:/Users/sungj/Desktop/야옹181014/지난/반출데이터csv/lstmdust5.csv', usecols=[0], engine='python', skipfooter=3)
#dataframe = read_csv('C:/Users/kroea/Desktop/sdndata/lstmdust.csv')

dataset = dataframe.values
dataset = dataset.astype('float32')
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
lookback = 10
trainX, testX = dataset[0:train_size-lookback,:], dataset[train_size:len(dataset)-lookback,:]
trainY, testY = dataset[0+lookback:train_size,:], dataset[train_size+lookback:len(dataset),:]

trainX
trainY
testX
testY



trainX = create_dataset(trainX, lookback)
trainY = create_dataset(trainY, lookback)
testX = create_dataset(testX , lookback)
testY = create_dataset(testY , lookback)
trainX = trainX.reshape(len(trainX), lookback, 1)
trainY = trainY.reshape(len(trainY), lookback, 1)
testX = testX.reshape(len(testX), lookback, 1)
testY = testY.reshape(len(testY), lookback, 1)
#y = trainY.reshape(1, lookback, 1)
# define LSTM configuration
n_neurons = lookback
n_batch = 1
n_epoch = 30
# create LSTM
model = Sequential()
#model.add(LSTM(1, input_shape=(1, 1), return_sequences=True))
#model.add(LSTM())
model.add(LSTM(n_neurons, input_shape=(lookback, 1), return_sequences=True,activation='relu'))
model.add(TimeDistributed(Dense(1)))
#model.add(TimeDistributed(Dense(1,activation='relu')))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
#model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

print(model.summary())
# train LSTM
model.fit(trainX, trainY, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(testX, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)

 
model.evaluate(x=(1,2,3,4),y=(3,4,5,6)) 



trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MAPE'  % trainScore)
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MAPE'  % testScore)
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


testX
testY
print(testPredict)
a=testPredict[-1:]
predict=a.reshape(lookback)
b=testY[-1:]
real=b.reshape(lookback)
#c=testPredict[1214:0]
plt.plot(predict)
plt.plot(real)
plt.plot(trainX[0:1793,0])
plt.plot(testX[0:1104,0])










# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot[1:])
plt.xlim(32500,33700)
#plt.ylim(-0.1,0.2)
#line1, = plt.plot(dataset[train_size:len(dataset),:], label='Test Y')
plt.show()
dataset


print(trainX)