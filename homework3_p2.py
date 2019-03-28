import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_operators
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc
import time # has the equivalent of tic/toc

machine_epsilon = np.finfo(float).eps

#########################################
############### User Input ##############

# number of (pressure) cells = mass conservation cells
Nxc  = 5
Nyc  = 5
Np   = Nxc*Nyc
Lx   = 2.*np.pi
Ly   = 2.*np.pi

#########################################
######## Preprocessing Stage ############

# You might have to include ghost-cells here
# Depending on your application

# define grid for u and v velocity components first
# and then define pressure cells locations
xsi_u = np.linspace(0.,1.0,Nxc+1)
xsi_v = np.linspace(0.,1.0,Nyc+1)
# uniform grid
xu = (xsi_u)*Lx
yv = (xsi_v)*Ly
# print(x_u)

# (non-sense non-uniform grid)
#xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
#yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid

# creating ghost cells
dxu0 = np.diff(xu)[0]
dxuL = np.diff(xu)[-1]
xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])

dyv0 = np.diff(yv)[0]
dyvL = np.diff(yv)[-1]
yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])

dxc = np.diff(xu)  # pressure-cells spacings
dyc = np.diff(yv)

xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells

# note that indexing is Xc[j_y,i_x] or Xc[j,i]
[Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
[Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells

# Pre-allocated at all False = no fluid points
pressureCells_Mask = np.zeros(Xc.shape)
pressureCells_Mask[1:-1,1:-1] = True

# number of actual pressure cells
Np = len(np.where(pressureCells_Mask==True)[0])
q  = np.ones(Np,)

DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask)

# Set delta_x and delta_t
dx = dxc
dt = 0.01


# Imposing initial conditions at t = 0
u = np.zeros((Nxc+1,Nyc+1))
v = np.zeros((Nxc+1,Nyc+1))
p = np.zeros((Nxc+1,Nyc+1))

for i in range(Nyc+1):
    for j in range(Nxc+1):
        u[Nyc-i][j] = (-np.cos(xu[i]) * np.sin(yv[j]))

for i in range(Nyc+1):
    for j in range(Nxc+1):
        v[Nxc-i][j] = (-np.sin(xu[i]) * np.cos(yv[j]))

for i in range(Nyc+1):
    for j in range(Nxc+1):
        p[Nxc-i][j] = (-(np.cos(2 * xu[i]) * np.sin(2 * yv[j])) / 4)

# Now matrix u v p contains the initial condition of corresponding coordinates


# Discretization
# u momentum discretization
u_star = np.zeros((Nxc+1,Nyc+1))
for i in range(Nyc+1):
    for j in range(Nxc+1):
        v_current = 0.25 * (v[i-1][j] + v[i-1][j+1] + v[i][j] + v[i][j+1])
        u_star[Nyc-i][j] = u[i][j] + dt * ((u[i][j] * (u[i-1][j] - 2 * u[i][j] + u[i+1][j]) / dx ** 2) + (u[i][j] * (u[i][j-1] - 2 * u[i][j] + u[i][j+1]) / dx ** 2) - u[i][j] * (u[i+1][j] - u[i-1][j] - u[i-1][j]) / (2 * dx) - v_current * (u[i][j+1] - u[i][j-1]) / (2 * dx))


# v momentum discretization
v_star = np.zeros((Nxc+1,Nyc+1))
for i in range(Nyc+1):
    for j in range(Nxc+1):
        u_current = 0.25 * (u[i-1][j] + u[i-1][j+1] + u[i][j] + u[i][j+1])
        v_star[Nyc-i][j] = v[i][j] + dt * ((v[i][j] * (v[i-1][j] - 2 * v[i][j] + v[i+1][j]) / dx ** 2) + (v[i][j] * (v[i][j-1] - 2 * v[i][j] + v[i][j+1]) / dx ** 2) - v[i][j] * (v[i+1][j] - v[i-1][j] - v[i-1][j]) / (2 * dx) - u_current * (v[i][j+1] - v[i][j-1]) / (2 * dx))

# Poissson equation

# solving pressure
R = np.zeros((Nxc+1)*(Nyc+1),1)
k = 1
for i in range(Nyc+1):
    for j in range(Nxc+1):
        R[k] = (u_star[i+1][j] - u_star[i][j]) / dx + (v_star[i+1][j] - v_star[i][j]) / dx
        k += 1

# update pressure
pv = np.zeros((Nxc+1)*(Nyc+1),1)
pv = np.linalg.solve(DivGrad, R)
for i in range(Nyc+1):
    for j in range(Nxc+1):
        p[Nyc-i][j] = pv[k]
        k += 1
# Pressure is obtained

# Corrector step
for i in range(Nyc+1):
    for j in range(Nxc+1):
        u[Nyc-i][j] = u_star[i][j] - dt * (p[i][j] - p[i-1][j]) / dx
for i in range(Nyc+1):
    for j in range(Nxc+1):
        v[Nyc-i][j] = v_star[i][j] - dt * (p[i][j] - p[i-1][j]) / dx
# u and v velocities are obtained

print(u)
print(v)
print(p)

