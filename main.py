data_path = "digit-recognizer/train.csv"

import tensorflow as tf
import numpy as np
import csv

x_train = []
y_train = []


# LOAD TRAIN DATA
with open(data_path, 'r') as data_file:
    csv_reader = csv.reader(data_file)
    line_count = 0
    for row in csv_reader:
        if(line_count != 0):
            y_train.append([row[0]])
            x_train.append(row[1:])
        line_count += 1

#reshape training data to (42000, 28, 28, 1)
x_train = np.array(x_train, dtype="int")
dims = int(np.sqrt(x_train.shape[1]))
x_train = np.reshape(x_train, (x_train.shape[0], dims, dims))
x_train = tf.expand_dims(x_train, axis=3)
y_train = np.array(y_train, dtype="int")

#Normalize training data
x_train = x_train / 255

print("Y shape: " + str(y_train.shape))
print("X shape: " + str(x_train.shape))

#time for model folder creation
import time
start_time = round(time.time())

#callback that saves the model on every epoch
import os
import pickle
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_weights = None
        self.best_weights_acc = 0
        self.best_model_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        print("epoch")
        if(logs.get('acc') > self.best_weights_acc):
            self.best_weights_acc = logs.get('val_acc')
            self.best_weights = self.model.get_weights()
            self.best_model_epoch = epoch

        foldername = "{}/models/{}/epoch{}-{}".format(os.getcwd(), start_time, epoch, logs.get('val_acc'))
        os.makedirs(foldername)
        with open('{}/current_weights.obj'.format(foldername), 'wb') as obj:
            pickle.dump(self.model.get_weights(), obj)

    def on_train_end(self, logs={}):
        print("end")
        foldername = "{}/models/{}/bestmodel-{}".format(os.getcwd(), start_time, self.best_model_epoch)
        os.makedirs(foldername)
        with open('{}/best_weights.obj'.format(foldername), 'wb') as obj:
            pickle.dump(self.best_weights, obj)

cp_callback = myCallback()

# get the model from modelBuild.py
from modelBuild import myModel
model = myModel.getModel()

#train the model
history = model.fit(x_train, y_train, validation_split=0.05, epochs=20, callbacks=[cp_callback])


#plot the loss, val loss, accuracy and val accuracy
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2)
axs[0].plot(history.history['acc'], label='accuracy', color="b")
axs[0].plot(history.history['val_acc'], label='val accuracy', color="g")
axs[1].plot(history.history['loss'], label='loss', color="b")
axs[1].plot(history.history['val_loss'], label='val loss', color="g")

plt.show()
