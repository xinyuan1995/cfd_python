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
Nxc  = 130
Nyc  = 130
Np   = Nxc*Nyc
Lx   = 1
Ly   = 1
Re = 100
n = 10
#########################################
######## Preprocessing Stage ############

# You might have to include ghost-cells here
# Depending on your application

# define grid for u and v velocity components first
# and then define pressure cells locations
xsi_u = np.linspace(0.,1.0,Nxc+1)
xsi_v = np.linspace(0.,1.0,Nyc+1)
xi_u = np.linspace(0.,1.0,Nxc).reshape(Nxc,1)
yi_v = np.linspace(0.,1.0,Nyc).reshape(Nyc,1)
# uniform grid
xu = (xsi_u)*Lx
yv = (xsi_v)*Ly
x_u = (xi_u)*Lx
y_v = (yi_v)*Ly
# (non-sense non-uniform grid)
# xu = (xsi_u**0.5)*Lx # can be replaced by non-uniform grid
# yv = (xsi_v**0.5)*Ly # can be replaced by non-uniform grid

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
print(x_u.shape)
print(y_v.shape)

### familiarize yourself with 'flattening' options
# phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
# phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
# phi_PythonDefault = Phi.flatten()          # Python default
# compare Phi[:,0] and Phi[0,:] with phi[:Nxc]

# Pre-allocated at all False = no fluid points
pressureCells_Mask = np.zeros(Xc.shape)
pressureCells_Mask[1:-1,1:-1] = True

# # Introducing obstacle in pressure Mask
# obstacle_radius = 0.3*Lx # set equal to 0.0*Lx to remove obstacle
# distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
# j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
# pressureCells_Mask[j_obstacle,i_obstacle] = False

# number of actual pressure cells
Np = len(np.where(pressureCells_Mask==True)[0])
q  = np.ones((Np,1))


# a more advanced option is to separately create the divergence and gradient operators
DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Dirichlet")
# DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask)
Divx = spatial_operators.Divx_operator(Dxc,Xc,pressureCells_Mask,boundary_conditions="Homogeneous Dirichlet")
# Divx = spatial_operators.Divx_operator(Dxc,Xc,pressureCells_Mask,boundary_conditions="Homogeneous Neumann")
# if boundary_conditions are not specified, it defaults to "Homogeneous Neumann"
plt.spy(DivGrad)
# plt.title('Spy plot for Homogeneous Neumann')
# plt.savefig('Spy plot for Homogeneous Neumann.png')
plt.show()
print(DivGrad.shape)


############# Successive over relaxation ##################
def fcn_q(Xc,Yc,n,Re):
    fcn_q = np.sin(2*np.pi*n*Yc)*(2*np.pi*n*np.cos(2*np.pi*n*Xc)+(8*n**2*np.pi**2/Re)*(np.sin(2*np.pi*n*Xc)))
    return fcn_q
fc = []
for i in range(Nxc):
    for j in range(Nyc):
        fc.append(fcn_q(x_u[i],y_v[j],n,Re))
fc = np.array(fc)

A = Divx - DivGrad / Re
A1 = scysparse.tril(A)
A2 = A1 - A
invA1 = spysparselinalg.inv(A1)
b = spysparselinalg.inv(A1) * fc


# Looking for the optimal omega
omegas = np.linspace(1,1.7,30);
ks = []
for omega in omegas:
    phi = np.zeros((Np,1))
    k = 0
    while (sum(sum(abs(fc - A*phi))) / sum(sum(abs(fc))) > 1e-5):
        k += 1
        phi_star = (np.eye(Np) - invA1 * A) * phi + b
        phi = omega * phi_star + (1 - omega) * phi
        # print(omega, k, sum(sum(abs(fc - A*phi))) / sum(sum(abs(fc))))
    ks.append(k)
    print(omega, k)
ks = np.asarray(ks)
# omegas.tofile('omega.dat')
# ks.tofile('k20.dat')
# plt.plot(omegas,ks)
plt.show()

# After getting the optimal omega
omega = 1.37
ks = []
rs = []
phi = np.zeros((Np,1))
k = 0

while (sum(sum(abs(fc - A*phi))) / sum(sum(abs(fc))) > 1e-5):
    k += 1
    phi_star = (scysparse.eye(Np) - invA1 * A) * phi + b
    phi = omega * phi_star + (1 - omega) * phi
    rk = sum(sum(abs(fc - A*phi))) / Np
    ks.append(k)
    rs.append(rk)
    # print(rk)
ks = np.asarray(ks)
rs = np.asarray(rs)
ks.tofile('k1e8_x.dat')
rs.tofile('k1e8_y.dat')
plt.plot(ks,rs)
plt.show()



######## Non-linear case a)
phi = np.zeros((Np, 1))
A = - DivGrad / Re
q = -np.multiply(phi, Divx * phi) + fc
k = 0
start_time = time.time()

while (sum(sum(abs(-np.multiply(phi, Divx * phi) + fc - A*phi))) / sum(sum(abs(-np.multiply(phi, Divx * phi) + fc))) > 1e-5):
    k += 1
    q = -np.multiply(phi, Divx * phi) + fc
    phi = spysparselinalg.spsolve(A,q) # solving nabla*phi = 1
    print(k)
print('end')
print("--- %s seconds ---" % (time.time() - start_time))


# Non-linear case b)
l = 0
ep = -2 / Re # regularization term
phi = np.zeros((Np, 1))
A = np.multiply(phi, Divx * phi) + ep * DivGrad * phi
q = DivGrad / Re + fc + ep * DivGrad * phi
start_time = time.time()

while (sum(sum(abs(DivGrad / Re + fc + + ep * DivGrad * phi - A*phi))) / sum(sum(abs(DivGrad / Re + fc + + ep * DivGrad * phi))) > 1e-5):
    k += 1
    q = DivGrad / Re + fc + ep * DivGrad * phi
    # sub iteration l
    while l < 101:
        phi += l / 100 * phi
        A = np.multiply(phi, Divx * phi) + ep * DivGrad * phi
    phi = spysparselinalg.spsolve(A, q)
    print(k)
print('end')
print("--- %s seconds ---" % (time.time() - start_time))

