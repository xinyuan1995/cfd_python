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

#Centered collocated arrangement l=r=1
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x, 0, delta_x])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**2
#     # print(Truncation_error)
#     print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Centered collocated arrangement l = r = 1')
# plt.loglog(plot_x,plot_z,'r--',label='n = 2')
# plt.legend()
# plt.savefig('Centered collocated arrangement l = r = 1.png')

#Centered collocated arrangement l=r=2
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x*2,-delta_x, 0, delta_x, delta_x*2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**4
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Centered collocated arrangement l = r = 2')
# plt.loglog(plot_x,plot_z,'r--',label='n = 4')
# plt.legend()
# plt.savefig('Centered collocated arrangement l = r = 2.png')
# plt.show()

#Centered collocated arrangement l=r=3
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x*3, -delta_x*2,-delta_x, 0, delta_x, delta_x*2, delta_x*3])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**6
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Centered collocated arrangement l = r = 3')
# plt.loglog(plot_x,plot_z,'r--',label='n = 6')
# plt.legend()
# plt.savefig('Centered collocated arrangement l = r = 3.png')
# plt.show()

#Biased collocated arrangement l=0, r=1
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([0, delta_x])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**1
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Biased collocated arrangement l = 0, r = 1')
# plt.loglog(plot_x,plot_z,'r--',label='n = 1')
# plt.legend()
# plt.savefig('Biased collocated arrangement l = 0,  r = 1.png')
# plt.show()

#Biased collocated arrangement l=0, r=2
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([0, delta_x, delta_x*2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**2
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Biased collocated arrangement l = 0, r = 2')
# plt.loglog(plot_x,plot_z,'r--',label='n = 2')
# plt.legend()
# plt.savefig('Biased collocated arrangement l = 0,  r = 2.png')
# plt.show()

#Biased collocated arrangement l=0, r=3
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([0, delta_x, delta_x*2, delta_x*3])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**3
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Biased collocated arrangement l = 0, r = 3')
# plt.loglog(plot_x,plot_z,'r--',label='n = 3')
# plt.legend()
# plt.savefig('Biased collocated arrangement l = 0,  r = 3.png')
# plt.show()

#Centered staggered arrangement l=r=1
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x/2, delta_x/2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**1
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Centered staggered arrangement l = r = 1')
# plt.loglog(plot_x,plot_z,'r--',label='n = 1')
# plt.legend()
# plt.savefig('Centered staggered arrangement l = r = 1.png')
# plt.show()

#Centered staggered arrangement l=r=2
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x*3/2, -delta_x/2, delta_x/2, delta_x*3/2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**3
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Centered staggered arrangement l = r = 2')
# plt.loglog(plot_x,plot_z,'r--',label='n = 3')
# plt.legend()
# plt.savefig('Centered staggered arrangement l = r = 2.png')
# plt.show()

#Centered staggered arrangement l=r=2
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x*5/2, -delta_x*3/2, -delta_x/2, delta_x/2, delta_x*3/2, delta_x*5/2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**5
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Centered staggered arrangement l = r = 3')
# plt.loglog(plot_x,plot_z,'r--',label='n = 5')
# plt.legend()
# plt.savefig('Centered staggered arrangement l = r = 3.png')
# plt.show()

#Biased staggered arrangement l=r=1
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x/2, delta_x/2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**1
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Biased staggered arrangement l = r = 1')
# plt.loglog(plot_x,plot_z,'r--',label='n = 1')
# plt.legend()
# plt.savefig('Biased staggered arrangement l = r = 1.png')
# plt.show()

#Biased staggered arrangement l=1, r=2
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x/2, delta_x/2, delta_x*3/2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**2
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Biased staggered arrangement l = 1, r = 2')
# plt.loglog(plot_x,plot_z,'r--',label='n = 2')
# plt.legend()
# plt.savefig('Biased staggered arrangement l = 1, r = 2.png')
# plt.show()

#Biased staggered arrangement l=1, r=3
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.9975
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([-delta_x/2, delta_x/2, delta_x*3/2, delta_x*5/2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     plot_x[i-1] = 1/delta_x
#     plot_y[i-1] = Truncation_error
#     plot_z[i-1] = delta_x**3
#     # print(Truncation_error)
#     # print(1/delta_x)
# plt.loglog(plot_x, plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('Truncation error')
# plt.title('Biased staggered arrangement l = 1, r = 3')
# plt.loglog(plot_x,plot_z,'r--',label='n = 3')
# plt.legend()
# plt.savefig('Biased staggered arrangement l = 1, r = 3.png')
# plt.show()