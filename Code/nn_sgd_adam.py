import numpy as np
import pandas as pd
import keras 
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn import preprocessing # for normalize and scale feature

# The program reads in feature and target csv files and convert them into numpy arrays.
# The NN contains 3 layers, with categorical crossentropy loss fn, SGD as optimizer
# the lower the validation loss, the better the model
# At the end, it plots number of epochs vs validation loss

#
# For mac plt setup only, if plt does not work uncomment this
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#

featurepath='/Users/minaxxan/Downloads/features.csv'
targetpath = '/Users/minaxxan/Downloads/target.csv'

# drop the first column, 'index' of feature and target
# convert pd df to 2d numpy array
feature=pd.read_csv(featurepath); feature.drop(feature.columns[0],inplace=True,axis=1)
#feature = preprocessing.normalize(feature)
feature= feature.values # only need this if we dont normalize feature df

target=pd.read_csv(targetpath); target.drop(target.columns[0],inplace=True,axis=1)
# one hot encode
target = to_categorical(target)

n_cols = feature.shape[1]
early_stopping_monitor = EarlyStopping(patience=3) # stop running epochs if val_loss does not improve


#
# 1) sgd stochastic gradient descent optimizer
model = Sequential()

# Add the 3 hidden layers
model.add(Dense(units=25,activation='relu',input_shape=(n_cols,)))
model.add(Dense(units=75,activation='relu'))
model.add(Dense(units=100,activation='relu'))

# Add the output layer
model.add(Dense(units=3,activation='softmax'))

# Compile the model
model.compile(optimizer=SGD(0.01),loss='categorical_crossentropy',
               metrics=['accuracy'])

# Fit the model, lower the val_loss score the better the model
mtrained = model.fit(feature,target,validation_split=0.3, epochs = 20, callbacks=[early_stopping_monitor])

# plot number of epochs vs validation loss vs validation accuracy
validation_loss_score = mtrained.history['val_loss']
plt.figure(1)
plt.plot(validation_loss_score)
plt.xlabel('epochs'); plt.ylabel('validation loss'); plt.title('SGD: validation loss vs epochs')





# adam shows better result at validation accuracy of 57%, training accuracy of 90%
# 2) adam optimizer
model2 = Sequential()

# Add the 3 layers
model2.add(Dense(units=25,activation='relu',input_shape=(n_cols,)))
model2.add(Dense(units=100,activation='relu'))
model2.add(Dense(units=100,activation='relu'))

# Add the output layer
model2.add(Dense(units=3,activation='softmax'))
# Compile the model
model2.compile(optimizer='adam',loss='categorical_crossentropy',
               metrics=['accuracy'])
# Fit the model, lower the val_loss score the better the model
m2trained = model2.fit(feature,target,validation_split=0.3, epochs = 20, callbacks=[early_stopping_monitor])

# plot number of epochs vs validation loss vs validation accuracy
validation_loss_score = m2trained.history['val_loss']
plt.figure(2)
plt.plot(validation_loss_score)
plt.xlabel('epochs'); plt.ylabel('validation loss'); plt.title('ADAM: validation loss vs epochs')
plt.show()
