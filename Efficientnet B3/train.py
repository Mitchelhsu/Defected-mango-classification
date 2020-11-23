from google.colab import drive

drive.mount('/content/drive')

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

label = pd.read_csv('/content/drive/MyDrive/Mango/cp_label.csv')

clas = ['Bad_color', 'Anthrax', 'Milk_A', 'Machine_harm', 'Black_spot']

conditions = label['Condition']
new_labeled = [clas.index(con) for con in conditions]

label['Condition'] = new_labeled

path = '/content/drive/MyDrive/Mango/cp_train/'
iter = [i for i in range(len(label))]
X = []
Y = []

cnt = 0
for i in random.sample(iter, len(iter)):
    img = cv2.imread(path+label.loc[i, 'Img'])
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = tf.io.gfile.GFile(path+label.loc[i, 'Img'], 'rb').read()
    #img = tf.image.decode_jpeg(img)
    #img = tf.image.resize(img, [300, 300], method=tf.image.ResizeMethod.BILINEAR)
    #res = keras.preprocessing.image.img_to_array(img)
    X.append(img)
    Y.append(label.loc[i, 'Condition'])

    cnt += 1
    clear_output(wait=True)
    perc = (cnt/len(iter))*100
    print('%.2f'%perc, '%')
    print(i, cnt)



print(len(X), len(Y))

X = np.array(X)
Y = np.array(Y)
for i in range(len(X)):
    X[i] = X[i].astype(float)

Y = keras.utils.to_categorical(Y, num_classes=5)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(len(x_train), len(x_test))

type(x_train[0])

from tensorflow.keras.applications import EfficientNetB3

# build model
inputs = layers.Input(shape=(300, 300, 3))
outputs = EfficientNetB3(include_top=True, weights=None, classes=5)(inputs)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# train model
batch_size = 16
epochs = 30
CB = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.2, callbacks = [CB], verbose=1)




