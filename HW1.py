import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

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
# delta_x = 1
x = np.array([-1, 0, 1])
y = np.array([-0.2672, 0, 0.1638])
Lagrangian_poly = scipy.interpolate.lagrange(x, y)
print(Lagrangian_poly)  # check the polynomial comment out when necessary
y_function =  np.tanh(x) * np.sin(5 * x + 1.5)
#Truncation_error = abs(y_function - Lagrangian_poly)
should be derivative, not the function
print(Truncation_error)
u = plt.plot(x[2], Truncation_error[2], 'ro')
plt.show()