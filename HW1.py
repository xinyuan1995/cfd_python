import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

x = np.array([-1, 0, 1])
y = np.array([-0.2672, 0, 0.1638])
plt.figure()
u = plt.plot(x,y,'ro')
t = np.linspace(0, 1, len(x))
x_interpolation = scipy.interpolate.lagrange(t, x)
y_interpolation = scipy.interpolate.lagrange(t, y)
#t_lagrange = np.linspace(t[0],t[-1],200)
t_lagrange = np.linspace(0,len(x),200)
x_lagrange = x_interpolation(t_lagrange)
y_lagrange = y_interpolation(t_lagrange)
plt.plot(x_lagrange, y_lagrange)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrangian polynomial for N=3')
plt.savefig('Lagrangian polynomial for N=3.png')
