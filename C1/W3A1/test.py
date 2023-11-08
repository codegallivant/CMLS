import numpy as np	
import matplotlib.pyplot as plt
# from numpy import array
# from numpy import arange
# from numpy import meshgrid


fig, al = plt.subplots()
delta = 0.025
xrange = np.arange(-5.0, 20.0, delta)
yrange = np.arange(-5.0, 20.0, delta)
X, Y = np.meshgrid(xrange,yrange)

w = np.array([0.83335924, 0.85726968, 0.8618935,  0.85944093, 0.93178418])
# F is one side of the equation, G is the other
F = (w[0]*X) + (w[1]*Y) + (w[2]*(X**2)) + (w[3]*(Y**2)) + (w[4]*X*Y)

al.contour(X, Y, F, [0])
plt.show()
# 