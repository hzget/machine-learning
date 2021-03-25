
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np

def prepare_data(arr):
    b = move_to_center(arr)
    b = blur_withnearest(b)
    b = gaussian_filter(b, sigma=0.2) 
    return b

def move_to_center(im_arr):
    ys, xs = convert_to_xy(im_arr)
    delta_x = xs - np.average(xs)
    delta_y = ys - np.average(ys)
    xs2 = 14 + delta_x 
    ys2 = 14 + delta_y 
    b = np.zeros(im_arr.shape)
    for x1, y1 in zip(xs2, ys2):
        if x1 > 27:
            x1 = 27
        if x1 < 0:
            x1 = 0
        if y1 > 27:
            y1 = 27
        if y1 < 0:
            y1 = 0
        b[int(y1),int(x1)] = 1
    return b

def blur_withnearest(input):
    output = np.zeros((28,28))
    for x in range(1,26,1):
        for y in range(1,26,1):
            if (input[x,y]>0):
                output[x,y] = input[x,y]
            else:
                output[x,y] = (input[x-1,y-1] + input[x,y-1] + input[x+1,y-1] + 
                               input[x-1,y]   + input[x,y]   + input[x+1,y]   +
                               input[x-1,y+1] + input[x,y+1] + input[x+1,y+1])/9
    return output
    
def convert_to_xy(im_arr):
    ys, xs = [], []
    for x1 in range(28):
        for y1 in range(28):
            if im_arr[y1, x1] > 0:
                xs.append(x1)
                ys.append(y1)
    return np.array(ys), np.array(xs)

def loadimage():
    size = 28, 28
    img = Image.open('pic/digit.png')
    img = img.resize(size, Image.NEAREST)
    im = np.array(img)
    im = im[:,:,0]
    b = im/255
    return b

