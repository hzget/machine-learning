# check the costs after every time consuming all examples
# usage: python check_costs.py

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
import numpy as np
import matplotlib.pyplot as plt

net = network.Network([784, 30, 10])
net.SGD(training_data, 5, 10, 3.0, test_data=test_data)

# draw a picture
xpoints = []
ypoints = []
for x, y in net.costs:
    xpoints.append(x)
    ypoints.append(y)

plt.plot(xpoints, ypoints, marker = 'o', mec = 'r', mfc = 'r')
plt.xlabel('# of input')
plt.ylabel('average cost')
plt.show()
