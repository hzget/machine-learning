# check the running time for one loop
# usage: python spantime.py

# import dataset
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network

spans = []
net = network.Network([784, 30, 10])

for i in range(1, 10):
    t = net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
    spans.append(t)

# print("running time: {0}".format(spans))
print(spans)
