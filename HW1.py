import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from sympy import *

# # Given points for interpolation
# x = np.array([-1, 0, 1])
# y = np.array([-0.2672, 0, 0.1638])
# u = plt.plot(x, y, 'ro')
# # Obtain the Lagrange polynomial of the given points
# Lagrangian_poly = scipy.interpolate.lagrange(x, y)
# print(Lagrangian_poly)  # check the polynomial comment out when necessary
# # Create the x axis line space as t
# t = np.linspace(-1, 1, 100)
# # Plot the polynomial
# ax = plt.plot(t, Lagrangian_poly(t))
# # Labels and title
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Lagrangian polynomial for N=3')
# plt.savefig('Lagrangian polynomial for N=3.png')


# x = np.zeros(41)
# y = np.zeros(41)
# for i in range(-20, 21):
#     y[i+20] = np.tanh(i)*np.sin(5*i+1.5)
#     x[i+20] = i
# u = plt.plot(x, y, 'ro')
# # Obtain the Lagrange polynomial of the given points
# Lagrangian_poly = scipy.interpolate.lagrange(x, y)
# print(Lagrangian_poly)  # check the polynomial comment out when necessary
# # Create the x axis line space as t
# t = np.linspace(-1, 1, 100)
# # Plot the polynomial
# ax = plt.plot(t, Lagrangian_poly(t))
# # Labels and title
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Lagrangian polynomial for N=41')
# plt.savefig('Lagrangian polynomial for N=41.png')

#Centered arrangement l=r=1
plot_y = np.zeros(101)
plot_x = np.zeros(101)
y_prime0 = 0.258819
for i in range (100, 0, -1):
    delta_x = i/100
    x = np.array([-delta_x, 0, delta_x])
    y = np.tanh(x) * np.sin(5*x + 1.5)
    Lagrangian_poly = scipy.interpolate.lagrange(x, y)
    Lag_prime = np.poly1d.deriv(Lagrangian_poly)
    Truncation_error = abs(y_prime0 - Lag_prime(0))
    # print(Truncation_error)
    plot_x[i] = 1/delta_x
    plot_y[i] = Truncation_error
    # plt.plot(plot_x, plot_y, 'ro')
    plt.loglog(plot_x,plot_y)
    plt.show()
    # u = plt.plot(x[2], Truncation_error[2], 'ro')
    # plt.show()
