 # Drawing a sin(x) graph.
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
'''
# How to use matplotlib
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots() # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) # Plot some data on the axes.
plt.show()
'''
'''
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np

fig = plt.figure()   # an empty figure with no axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig, axs = plt.subplots(2, 2) # afigure with 2*2 grid of axes
# a figure with one axes on the left, and two on the right:
fig, axs = plt.subplot_mosaic([['left', 'right-top'], ['left', 'right-bottom']])
plt.show()                                                       
'''
