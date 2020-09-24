# check the running time for one loop
# usage: python spantime.py

# import dataset
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

spans = []
net = network.Network([784, 30, 10])

for i in range(1, 10):
    t = net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
    spans.append(t)

print(spans, np.mean(spans))

# draw a picture
red_patch = mpatches.Patch(color='red', label='Average {0}'.format(np.mean(spans)))
plt.plot(spans)
plt.xlabel('# of running')
plt.ylabel('time span')
plt.legend(handles=[red_patch])
plt.show()
