import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import mnist
import imageprepare as imp

def init_axes(ax, title):
    ax.cla()
    ax.set_xlim(0, 28)
    ax.set_ylim(28, 0)
    ax.set_title(title)
    return ax

fig = plt.figure()
ax = plt.subplot(221)
init_axes(ax, "write a digit")

ax2 = plt.subplot(222)
init_axes(ax2, "preprocessed image")

model = mnist.load_model()

x,y = [], []
a = np.zeros((28,28))
prepared = a.copy()
# create empty plot
points, = ax.plot([], [], 'o')
# ax.grid(True)

# cache the background
background = fig.canvas.copy_from_bbox(ax.bbox)

is_released = True

class btaction:

#    def train_model(self,event):
#        global net, btrain 
#        btrain.set_active(False)
#        net = network.train_model(epochs=3, mini_batch_size=100, eta=3.0)
#        btrain.set_active(True)

    def print_predict(self, event):
        global prepared
        p = predict(prepared)
        print("scores of the digits 0~9:\n {}".format(p))
        print("most likely to be: {}".format(np.argmax(p)))
        clear_axis4()
        s1 = "scores of the digits 0~9:"
        s2 = "{}\n".format(p)
        s3 = "most likely to be: {}".format(np.argmax(p))
        plt.text(0.1, 0.9, s1, fontsize=9, weight='bold')
        plt.text(0.1, -0.1, s2, fontsize=9)
        plt.text(0.1, -0.2, s3, fontsize=15, weight='bold', color='r')
    
    def plot_prediction(self, event):
        global prepared
        clear_axis4()
        ax4 = plt.subplot(2,2,4)
#        plt.axis('on')
        plotscoreimg(scores=predict(prepared), ax=ax4)
    
    def clear_xypoints(self, event):
        global x,y,a,ax2, prepared
        x,y = [],[]
        a = np.zeros((28,28))
        prepared = np.zeros((28,28))
        init_axes(ax2, "preprocessed image")
        reset_axis4()
        fresh_img(x,y)
    
    def preprocess(self,event):
        global a, prepared
        prepared = imp.prepare_data(a)
        reset_axis4()
        plotimg(prepared)
    
    def load_image(self,event):
        global x, y, a
        b = imp.loadimage()
        ys, xs = imp.convert_to_xy(b)
        y, x = ys.tolist(), xs.tolist()
        a = b
        fresh_img(x,y)

def clear_axis4():
    plt.subplot(2,2,4)
    plt.cla()
#    plt.axis('off')

def reset_axis4():
    plt.subplot(2,2,4)
    plt.cla()
    X = np.arange(10)
    scores = np.ones(10)*0.5
    bar = plt.bar(X.astype(str), scores.reshape(-1), fill=False, ls=':', label='?')
    plt.bar_label(bar, labels=['?']*10, label_type='center')
    xlabel = "most likely to be: {}".format("?")
    plt.xlabel(xlabel, fontsize=15, weight='bold', c='gray')
    plt.ylim(0, 1.1)

def plotimg(im_arr):
    ax2 = plt.subplot(222)
    plt.imshow(im_arr, cmap='gray')

def plotscoreimg(scores, ax):
    X = np.arange(10)
    digit = np.argmax(scores)
    ax.bar(X.astype(str), scores.reshape(-1))
#    ax.set_title("scores of 0~9")
    xlabel = "most likely to be: {}".format(digit)
    ax.set_xlabel(xlabel, weight='bold', color='r')
    ax.set_ylabel("scores of 0~9")
    ax.set_ylim(0, 1.1)

def predict(im_arr):
    global model
    p = model.predict(im_arr.reshape(1,28,28,1))
    return p[0]

def on_press(event):
#    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#          ('double' if event.dblclick else 'single', event.button,
#           event.x, event.y, event.xdata, event.ydata))
    global is_released, ax
    if event.inaxes is ax:
        is_released = False

def on_release(event):
    global is_released
    is_released = True

def on_move(event):
    global is_released, a, ax
    if is_released :
        return
    if event.inaxes is not ax:
        return
    # append event's data to lists
    x.append(event.xdata)
    y.append(event.ydata)
    a[int(event.ydata),int(event.xdata)]=1.0
    fresh_img(x,y)

def fresh_img(x,y):
    # update plot's data  
    points.set_data(x,y)
    # restore background
    fig.canvas.restore_region(background)
    # redraw just the points
    ax.draw_artist(points)
    # fill in the axes rectangle
    fig.canvas.blit(ax.bbox)

fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)

callback = btaction()
axpreprocess = plt.axes([0.2, 0.35, 0.2, 0.075])
bpreprocess = Button(axpreprocess, 'preprocess')
bpreprocess.on_clicked(callback.preprocess)

axpredict = plt.axes([0.2, 0.25, 0.2, 0.075])
bpredict= Button(axpredict, 'predict')
bpredict.on_clicked(callback.plot_prediction)

axclear = plt.axes([0.2, 0.15, 0.2, 0.075])
bclear = Button(axclear, 'clear')
bclear.on_clicked(callback.clear_xypoints)

#axtrain = plt.axes([0.2, 0.05, 0.2, 0.075])
#btrain = Button(axtrain, 'train model')
#btrain.on_clicked(callback.train_model)

#axloadimage = plt.axes([0.2, 0.05, 0.2, 0.075])
#bloadimage = Button(axloadimage, 'load digit')
#bloadimage.on_clicked(callback.load_image)

reset_axis4()

plt.show()