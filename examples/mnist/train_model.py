import mnist

mnn = mnist.mnist_nn()
mnn.prepare_data()
mnn.train_model()
mnn.save_model()
