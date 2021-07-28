import argparse

parser = argparse.ArgumentParser(description='train and save the model')
parser.add_argument('-m', '--model_file', dest='filename',
                    type=str, default='my_keras_model.h5',
                    help='the file to save the model. default file is my_keras_model.h5')
args = parser.parse_args()

filename = args.filename

import mnist

mnn = mnist.mnist_nn()
mnn.prepare_data()
mnn.train_model()

print(f'Save model to file {filename}')
mnn.save_model(filename)
