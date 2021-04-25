# python test_image.py
#  it will load image in 'pic/digit.png'
#  and load default model from "my_model.pkl"
#  and then predict digit of image

import numpy as np
import network
import imageprepare as imp

net = network.load_model()
im = imp.loadimage()
im = imp.prepare_data(im)

scores = net.predict(im.reshape(784,1))

print("scores of the digits 0~9:\n {}".format(scores))
print("most likely to be: {}".format(np.argmax(scores)))