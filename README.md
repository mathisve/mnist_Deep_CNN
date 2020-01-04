# mnist_Deep_CNN
Training a Deep Neural Network on the mnist dataset.

## Model architecture
```
Conv Layer (32 fiters, 5x5 size)
Conv Layer (32 fiters, 5x5 size)
MaxPooling (2x2 size)
Dropout (20%)

Conv Layer (64 filters, 3x3 size)
Conv Layer (128 filters, 2x2 size)
MaxPooling (2x2 size)
Dropout (15%)

Flatten
Dense (256 neurons, relu ativation)
Dense (10 neurons, softmax activation)
```
This was just an experiment to see if a DCNN is better suited for the mnist dataset than a regular shallow CNN. In the end it turns out it's not that efficient because the mnist dataset isn't that complex. From other experiments I have found it performs as similar to a very shallow neural network with only 1 or 2 conv layers. As you see in the picture below, I ended up overfitting after the 6th of 7th epoch.
I uploaded it's predictions of `test.csv`to Kaggle and got an accuracy of `98.942%` which is good, but not fantastic.
![plot image](https://github.com/Mathisco-01/mnist_Deep_CNN/blob/master/plot.png?raw=true)
