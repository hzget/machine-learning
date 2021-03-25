# check the costs after every time consuming all examples
# usage: python test_image.py

import network
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import imageprepare as imp

net = network.load_model()

size = 28, 28
img = Image.open('pic/digit.png')
#img = img.resize(size, Image.ANTIALIAS)
img = img.resize(size, Image.NEAREST)
im = np.array(img)
im = im[:,:,0]
b = im/255
b = imp.prepare_data(b)
p = net.feedforward(b.reshape(784,1))
print("scores of the digits 0~9:\n {}".format(p))
print("most likely to be: {}".format(np.argmax(p)))
plt.imshow(b, cmap='gray')
d = np.argmax(p)
plt.title('the below image is predicted as {}'.format(d), color = 'r')
plt.show()