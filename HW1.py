import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sympy import *

##############################
##############################
#Problem 1

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


# x = np.zeros(7)
# y = np.zeros(7)
# for i in range(-3, 4):
#     y[i+3] = np.tanh(i)*np.sin(5*i+1.5)
#     x[i+3] = i
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
# plt.title('Lagrangian polynomial for N=7')
# plt.savefig('Lagrangian polynomial for N=7.png')

#Centered collocated arrangement l=r=1
# plot_y = np.zeros(1000)
# plot_x = np.zeros(1000)
# plot_z = np.zeros(1000)
# y_prime0 = 0.99749498660405443094172337
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
#     print(Truncation_error)
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
# y_prime0 = 0.99749498660405443094172337
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
#     print(Truncation_error)
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
# for i in range(1000, 0, -1):
#     delta_x = i/1000
#     x = np.array([0, delta_x, delta_x*2])
#     y = np.tanh(x) * np.sin(5*x + 1.5)
#     Lagrangian_poly = scipy.interpolate.lagrange(x, y)
#     Lag_prime = np.poly1d.deriv(Lagrangian_poly)
#     Truncation_error = abs(y_prime0 - Lag_prime(0))
#     print(Lagrangian_poly)
#     print(Lag_prime)
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
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
#     print(Truncation_error)
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
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
# y_prime0 = 0.99749498660405443094172337
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


######################################
######################################
#Problem 2
#1st derivative, 2nd order
# a = [[0 for row in range(10)] for col in range(10)]
# for i in range(10):
#     for j in range(10):
#         if i == 0 and j < 3:
#             a[i][j] = 1
#         elif i < 9 and i > 0:
#             a[i][i-1] = 1
#             a[i][i] = 0
#             a[i][i+1] = 1
#         elif i == 9 and j > 6:
#             a[i][j] = 1
#         else:
#             a[i][j] = 0
# plt.spy(a, markersize=25)
# plt.title('Spy plot for 1st derivative, 2nd order')
# plt.savefig('Spy plot for 1st derivative, 2nd order.png')
# plt.show()

# #1st derivative, 3rd order
# a = []
# for i in range(10):
#     a.append([])
#     for j in range(10):
#         if j >= i and j < i + 4 and i < 7:
#             a[i].append(1)
#         elif i >=7 and j > 5:
#             a[i].append(1)
#         else:
#             a[i].append(0)
# plt.spy(a,markersize=25)
# plt.title('Spy plot for 1st derivative, 3rd order')
# plt.savefig('Spy plot for 1st derivative, 3rd order.png')
# plt.show()

# #2nd derivative, 2nd order
# a = []
# for i in  range(10):
#     a.append([])
#     for j in range(10):
#         if i == 0 and j < 4:
#             a[i].append(1)
#         elif i == 9 and j > 5:
#              a[i].append(1)
#         elif j >= i-1 and j < i + 2 and i < 8:
#             a[i].append(1)
#         elif i >=8 and j > 6:
#             a[i].append(1)
#         else:
#             a[i].append(0)
# plt.spy(a,markersize=25)
# plt.title('Spy plot for 2nd derivative, 2nd order')
# plt.savefig('Spy plot for 2nd derivative, 2nd order.png')
# plt.show()

#2nd derivative, 3rd order
# a = []
# for i in range(10):
#     a.append([])
#     for j in range(10):
#         if j >= i and j < i + 5 and i < 6:
#             a[i].append(1)
#         elif i >=6 and j > 4:
#             a[i].append(1)
#         else:
#             a[i].append(0)
# plt.spy(a,markersize=25)
# plt.title('Spy plot for 2nd derivative, 3rd order')
# plt.savefig('Spy plot for 2nd derivative, 3rd order.png')
# plt.show()

##############################
# Problem 2b)

# def fx(x):
#     return np.sin(x)
# def fpx(x):
#     return np.cos(x)
# def fpx_fppx(x):
#     return np.cos(x)-np.sin(x)
# plot_x = np.zeros(999)
# plot_y = np.zeros(999)
# plot_z = np.zeros(999)
# for index in range(999, 0, -1):
#     delta_x = index/1000
#     n = int(5/delta_x)
#     a = [[0 for row in range(n)] for col in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i == 0:
#                 if j == 0:
#                     a[i][j] = 1.
#             elif i < n-2:
#                 if j == i-1:
#                     a[i][j] = 1./np.power(delta_x,2.) - 1./(2.*delta_x)
#                 elif j ==i:
#                     a[i][j] = -2./np.power(delta_x,2.)
#                 elif j ==i+1:
#                     a[i][j] = 1./np.power(delta_x,2.) + 1./(2.*delta_x)
#             elif i == n-2:
#                 if j == n-3:
#                     a[i][j] = 1./np.power(delta_x,2.) - 1./(2.*delta_x)
#                 if j == n-2:
#                     a[i][j] = - 2./np.power(delta_x,2.)
#                 elif j == n-1:
#                     a[i][j] = 1./np.power(delta_x,2.) + 1./(2.*delta_x)
#             elif i == n-1:
#                 if j == n-3:
#                     a[i][j] = 1./(2.*delta_x)
#                 elif j == n-2:
#                     a[i][j] = -2./delta_x
#                 elif j == n-1:
#                     a[i][j] = 3./(2.*delta_x)
#             else:
#                 a[i][j] = 0.
#
#
#     b = [[0] for col in range(n)]
#     for i in range(n):
#         if i == 0:
#             b[i][0] = fx(0)
#         elif i == n-1:
#             b[i][0] = fpx(delta_x*i)
#         else:
#             b[i][0] = fpx_fppx(delta_x*i)
#     ainv = inv(a)
#     npb = np.array(b)
#     npainv = np.array(ainv)
#     res = np.dot(npainv, npb)
#     abssum = 0
#     func = [[0] for col in range(n)]
#     for i in range(n):
#         abssum += ((fx(i*delta_x) - res[i])**2)
#     plot_y[index - 1] = np.sqrt(abssum/n)
#     plot_x[index-1] = 1/delta_x
#     plot_z[index - 1] = delta_x ** 2
# plt.loglog(plot_x,plot_y)
# plt.loglog(plot_x,plot_z,'r--',label='n = 2')
# plt.legend()
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('RMS error')
# plt.title('2nd order accuracy scheme')
# plt.savefig('2nd order accuracy scheme.png')
# plt.show()


## 3rd order scheme
# def fx(x):
#     return np.sin(x)
# def fpx(x):
#     return np.cos(x)
# def fpx_fppx(x):
#     return np.cos(x)-np.sin(x)
# plot_x = np.zeros(999)
# plot_y = np.zeros(999)
# plot_z = np.zeros(999)
# for index in range(999, 0, -1):
#     delta_x = index/1000
#     n = int(5/delta_x)
#     a = [[0 for row in range(n)] for col in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i == 0:
#                 if j == 0:
#                     a[i][j] = 1.
#             elif i == n - 3:
#                 if j == n - 5:
#                     a[i][j] = -1. / (12. * delta_x ** 2)
#                 elif j == n - 4:
#                     a[i][j] = 16. / (12. * delta_x ** 2) - 1. / (3. * delta_x)
#                 elif j == n - 3:
#                     a[i][j] = -30. / (12. * delta_x ** 2) - 1. / (2. * delta_x)
#                 elif j == n - 2:
#                     a[i][j] = 16. / (12. * delta_x ** 2) + 1. / (1. * delta_x)
#                 elif j == n - 1:
#                     a[i][j] = -1. / (12. * delta_x ** 2) - 1. / (6. * delta_x)
#             elif i == n - 2:
#                 if j == n - 5:
#                     a[i][j] = -1. / (12. * delta_x ** 2)
#                 elif j == n - 4:
#                     a[i][j] = 4. / (12. * delta_x ** 2) + 1. / (6. * delta_x)
#                 elif j == n - 3:
#                     a[i][j] = 6. / (12. * delta_x ** 2) - 1. / (1. * delta_x)
#                 elif j == n - 2:
#                     a[i][j] = -20. / (12. * delta_x ** 2) + 1. / (2. * delta_x)
#                 elif j == n - 1:
#                     a[i][j] = 11. / (12. * delta_x ** 2) + 1. / (3. * delta_x)
#             elif i == n - 1:
#                 if j == n - 4:
#                     a[i][j] = -1. / (3. * delta_x)
#                 elif j == n - 3:
#                     a[i][j] = 3. / (2. * delta_x)
#                 elif j == n - 2:
#                     a[i][j] = -3. / (1. * delta_x)
#                 elif j == n - 1:
#                     a[i][j] = 11. / (6. * delta_x)
#             elif i > 0 and i < n - 3:
#                 if j == i - 1:
#                     a[i][j] = 11. / (12. * delta_x ** 2)
#                 elif j == i:
#                     a[i][j] = -20. / (12. * delta_x ** 2) - 11. / (6. * delta_x)
#                 elif j == i + 1:
#                     a[i][j] = 6. / (12. * delta_x ** 2) + 3. / (1. * delta_x)
#                 elif j == i + 2:
#                     a[i][j] = 4. / (12. * delta_x ** 2) - 3. / (2. * delta_x)
#                 elif j == i + 3:
#                     a[i][j] = -1. / (12. * delta_x ** 2) + 1. / (3. * delta_x)
#             else:
#                 a[i][j] = 0.
#
#
#     b = [[0] for col in range(n)]
#     for i in range(n):
#         if i == 0:
#             b[i][0] = fx(0)
#         elif i == n-1:
#             b[i][0] = fpx(delta_x*i)
#         else:
#             b[i][0] = fpx_fppx(delta_x*i)
#     ainv = inv(a)
#     npb = np.array(b)
#     npainv = np.array(ainv)
#     res = np.dot(npainv, npb)
#     abssum = 0
#     func = [[0] for col in range(n)]
#     for i in range(n):
#         abssum += ((fx(i*delta_x) - res[i])**2)
#     plot_y[index - 1] = np.sqrt(abssum/n)
#     plot_x[index-1] = 1/delta_x
#     plot_z[index - 1] = delta_x ** 3
# plt.loglog(plot_x,plot_y)
# plt.loglog(plot_x,plot_z,'r--',label='n = 3')
# plt.legend()
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('RMS error')
# plt.title('3rd order accuracy scheme')
# plt.savefig('3rd order accuracy scheme.png')
# plt.show()

#################################
#################################
#Problem 3
# def fx(x):
#     return np.power(x,3)
# def fpx(x):
#     return 3*np.power(x,2)
# def fppx(x):
#     return 6*x
# plot_x = np.zeros(999)
# plot_y = np.zeros(999)
# plot_z = np.zeros(999)
# for index in range(999, 0, -1):
#     delta_x = index/1000
# # delta_x =0.1
#     n = int(1/delta_x)
#     a = [[0 for row in range(n)] for col in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i == 0 and j == 0:
#                 a[i][j] = 1
#             elif i ==n-1 and j ==n-1:
#                 a[i][j] = 1
#             elif i > 0 and i < n-1:
#                 a[i][i-1] = 1/np.power(delta_x,2)
#                 a[i][i] = -2/np.power(delta_x,2)
#                 a[i][i+1] = 1/np.power(delta_x,2)
#             else:
#                 a[i][j] = 0
#
#
#
#     b = [[0] for col in range(n)]
#     for i in range(n):
#         if i == 0:
#             b[i][0] = 0
#         elif i == n-1:
#             b[i][0] = 1
#         else:
#             b[i][0] = 6*(delta_x*(i+1))
#
#
#
#     ainv = inv(a)
#
#     print(np.matrix(a))
#     print(np.matrix(b))
#
#     npb = np.array(b)
#     npainv = np.array(ainv)
#     res = np.dot(npainv, npb)
#     abssum = 0
#     func = [[0] for col in range(n)]
#     for i in range(n):
#         abssum += ((fx(i*delta_x) - res[i])**2)
#     plot_y[index - 1] = np.sqrt(abssum/n)
#     plot_x[index-1] = 1/delta_x
#     plot_z[index - 1] = delta_x ** 3
#
#
#
#
# plt.loglog(plot_x,plot_y)
# plt.loglog(plot_x,plot_z,'r--',label='n = 3')
# plt.legend()
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('RMS error')
# plt.title('2nd order accuracy scheme')
# plt.savefig('2nd order accuracy scheme for problem 3.png')
# plt.show()


