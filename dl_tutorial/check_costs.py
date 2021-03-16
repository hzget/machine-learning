# check the costs after every time consuming all examples
# usage: python check_costs.py

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
import numpy as np
import matplotlib.pyplot as plt

net = network.Network([784, 30, 10])
net.SGD(training_data, 3, 10, 3.0, test_data=test_data)
print("len of costs is {0}".format(len(net.costs)))

# draw a picture
plt.plot(net.costs, marker = 'o')
plt.xlabel('# of running')
plt.ylabel('average cost')
plt.show()
