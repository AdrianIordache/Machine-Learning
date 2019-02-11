
# coding: utf-8

# In[ ]:


import time
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


NAME = "cats-vs-dogs-CNN"

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0 # Normalize

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss = "binary_crossentropy", 
              optimizer = "adam", 
              metrics = ['accuracy'])

model.fit(X, y, batch_size = 32, epochs = 3, validation_split = 0.1, callbacks = [tensorboard])

model.save('64x3-CNN.model')

