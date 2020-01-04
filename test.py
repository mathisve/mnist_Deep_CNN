import tensorflow as tf
import numpy as np
import csv
from tqdm import tqdm
from modelBuild import myModel
import pickle

data_path = "digit-recognizer/test.csv"

x_train = []
with open(data_path, 'r') as data_file:
    csv_reader = csv.reader(data_file)
    line_count = 0
    for row in csv_reader:
        if(line_count != 0):
            r = []
            for item in row:
                r.append(int(item))
            x_train.append(r)
        else:
            line_count += 1

x_train = np.array(x_train)
print(x_train.shape)

dims = int(np.sqrt(x_train.shape[1]))
x_train = np.reshape(x_train, (x_train.shape[0], dims, dims))


x_train = tf.expand_dims(x_train, axis=3)
x_train = x_train / 255

print(x_train.shape)
print(x_train[0].shape)
print(x_train[0])

with open('best_weights.obj', 'rb') as obj:
    weights = pickle.load(obj)

model = myModel.getModel()
model.set_weights(weights)
print("PREDICTING NOW!!!")

preds = model.predict(x_train)
with open('preds.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ImageId","Label"])
    c = 1
    for i in preds:
        writer.writerow([c, np.argmax(i)])
        c += 1
