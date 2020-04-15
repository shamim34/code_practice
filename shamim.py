import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
os.chdir('directory name')
data_directory = 'data/'
data = []
for file in os.listdir(data_directory):
    if(str(file).split('.')[1] == 'csv'):
        reader = pd.read_csv(os.path.join(data_directory,file))
        reader = reader.dropna()
        data.append(reader.iloc[:,1:5].values.astype('float32'))
sz = []
sz.append(len(data[0]))
dat= data[0]
for i in range(1,len(data)):
    sz.append(len(data[i]))
    dat = np.vstack((dat,data[i]))
    print(dat.shape, sz[i])

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
dat[:] = sc.fit_transform(dat[:])
final_data = []
koto = 0
for i in sz:
    final_data.append(dat[koto: koto + i])
    koto += i
look_back = 7
data_x = []
data_y = []
for i in final_data:
    for j in range(len(i) - look_back):
        data_x.append(i[j: j + look_back])
        data_y.append(i[j + look_back])
data_x = np.array(data_x)
data_y = np.array(data_y)
x_train = data_x[0:500]
x_test = data_x[500:]
y_train = data_y[0:500]
y_test = data_y[500:]
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(x_test, y_test), verbose=2, shuffle=False)
yhat = model.predict(x_test)
res = sc.inverse_transform(yhat)
ori = sc.inverse_transform(y_test)
for i in range(len(x_test)):
    print(res[i], ori[i])


print(yhat)

































    

