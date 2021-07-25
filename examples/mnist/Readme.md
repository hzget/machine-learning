# Handwritten digit prediction

The dataset is from keras build-in MNist dataset

## applications

train_model.py -- load local mnist dataset, train it
and then save the mode to a file

test_online.py -- load local model and predict
on-line handwritten digit

## libs

mnist_loader.py -- load keras build-in dataset and save
it in a specific format in a file

mnist.py -- prepare data, train model, load model and so on

imageprepare.py -- utils used by test_online.py
