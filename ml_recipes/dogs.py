import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

# let's say height is normally distributed
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# the histogram shows the number of each type of dogs with a given height 
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'], histtype='bar')
plt.show()