import math
import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_discretization
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc

#################
#Problem 1
x = np.float128(math.pi)
y = np.float128(1./math.pi)
r = range(1,1000)
e = abs(math.pi - np.float64(math.pi * np.power(x,r) * np.power(y,r)))
plt.plot(e)
plt.xlabel('Number of multiplications')
plt.ylabel('Round-off error')
plt.title('Float128')
# plt.savefig('Float128.png')
plt.show()
##################

#Problem 2
machine_epsilon = np.finfo(float).eps

### this example script is hard-coded for periodic problems

#########################################
############### User Input ##############
def u_initial(X):
    return np.sin(0.2*np.pi*X-1)-np.cos(0.2*np.pi*X-1) # lambda functions are better..
def u_analytical(X):
    return np.exp(-0.04*np.pi**2*2.533) * (np.sin(0.2*np.pi*(X-2.533)-1) - np.cos(0.2*np.pi*(X-2.533)-1))
Nx  = 10
Lx  = 10
c_x_ref = 1.0
c_x   = 1.*c_x_ref  # (linear) convection speed
alpha = 1.0     # diffusion coefficients
plot_x = []
plot_y = []
# Tf  = Lx/(c_x_ref+machine_epsilon) # one complete cycle
Tf = 2.533
# At time = 0.175, the max value reached 0.6
CFL = 0.1 # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)
# while CFL < 0.3:
#       CFL += 0.001
#       print(CFL)
#       plot_every  = 1
# while Nx < 10000:
#       Nx += 100
#       print(Nx)
plot_every  = 1


## Time Advancement
time_advancement = "Explicit-Euler"
# time_advancement = "Crank-Nicolson"

## Advection Scheme
advection_scheme = "1st-order-upwind"
# advection_scheme = "2nd-order-central"
# advection_scheme = "2nd-order-upwind" #, QUICK

## Diffusion Scheme
diffusion_scheme = "2nd-order-central" # always-second-order central

ymin = -1.
ymax = 1.

######## Preprocessing Stage ############

# xx = np.linspace(0.,Lx,Nx+1)
# actual mesh points are off the boundaries x=0, x=Lx
# non-periodic boundary conditions created with ghost points
# x_mesh = 0.5*(xx[:-1]+xx[1:])
# dx  = np.diff(xx)[0]
# dx2 = dx*dx
C_c = 0.01
C_alpha = 0.01
while C_c < 2:
    C_c += 0.01
    while C_alpha < 2:
        C_alpha += 0.01
        dx = C_c / C_alpha
        dx2 = dx * dx
        dt = dx * C_c
        xx = np.linspace(0.,10, int(10/dx))
        x_mesh = 0.5*(xx[:-1]+xx[1:])

# for linear advection/diffusion time step is a function
# of c,alpha,dx only; we use reference limits, ballpark
# estimates for Explicit Euler
# dt_max_advective = dx/(c_x+machine_epsilon)             #   think of machine_epsilon as zero
# dt_max_diffusive = dx2/(alpha+machine_epsilon)
# dt_max = np.min([dt_max_advective,dt_max_diffusive])
# dt = CFL*dt_max
# dt = 0.01
# unitary_float = 1.+0.1*machine_epsilon # wat ?!


# Creating identity matrix
        Ieye = scysparse.identity(Nx)

# Creating first derivative
        Dx = spatial_discretization.Generate_Spatial_Operators(\
                x_mesh,advection_scheme,derivation_order=1)
# Creating second derivative
        D2x2 = spatial_discretization.Generate_Spatial_Operators(\
                x_mesh,diffusion_scheme,derivation_order=2)

# Creating A,B matrices such that:
#     A*u^{n+1} = B*u^{n} + q
        if time_advancement=="Explicit-Euler":
            A = Ieye
            B = Ieye-dt*c_x*Dx+dt*alpha*D2x2
        if time_advancement=="Crank-Nicolson":
            adv_diff_Op = -dt*c_x*Dx+dt*alpha*D2x2
            A = Ieye-0.5*adv_diff_Op
            B = Ieye+0.5*adv_diff_Op

#plt.spy(Dx)
#plt.show()

# forcing csr ordering..
        A , B = scysparse.csr_matrix(A),scysparse.csr_matrix(B)
# plt.spy(A)
# plt.title('Crank Nicholson, 1st-order Upwind scheme, A')
# plt.savefig('Crank Nicholson, 1st-order Upwind scheme, A.png')
# plt.show()

#########################################
####### Eigenvalue an# alysis #############
        T = (scylinalg.inv(A.todense())).dot(B.todense())  # T = A^{-1}*B
# plt.contourf(T)
# plt.show()
#         lambdas,_ = scylinalg.eig(T); plt.plot(np.abs(lambdas)); plt.show()
        Evalue = max(np.linalg.eigvals(T))
        print(Evalue)
# keyboard()

#########################################
########## Time integration #############

      # u = u_initial(x_mesh) # initializing solution
      # u_ana = u_analytical(x_mesh)

# Figure settings
#matplotlibrc('text.latex', preamble='\usepackage{color}')
#matplotlibrc('text',usetex=True)
#matplotlibrc('font', family='serif')

# figwidth       = 10
# figheight      = 6
# lineWidth      = 4
# textFontSize   = 28
# gcafontSize    = 30
#
# plt.ion()      # pylab's interactive mode-on
# plt.close()

      # time = 0.
      # it   = 0

# Plot initial conditions
# fig = plt.figure(0, figsize=(figwidth,figheight))
# ax   = fig.add_axes([0.15,0.15,0.8,0.8])
# plt.axes(ax)
# ax.plot(x_mesh,u_initial(x_mesh),'--k',linewidth=1)
# ax.text(0.7,0.9,r"$t="+"%1.5f" %time+"$",fontsize=gcafontSize,transform=ax.transAxes)
# ax.grid('on',which='both')
# plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
# plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
# ax.set_xlabel(r"$x$",fontsize=textFontSize)
# ax.set_ylabel(r"$u(x,t)$",fontsize=textFontSize,rotation=90)
# ax.set_ylim([ymin,ymax])
# plt.draw()
#
# _ = input('Plotting Initial Conditions. Press Key to start time integration')
# plt.close()
# plt.ion()

#       while time < Tf:
#
#          it   += 1
#          time += dt
#
#    # print("Iteration = " + repr(it))
#
#
#    # Update solution
#    # solving : A*u^{n+1} = B*u^{n} + q
#    # where q is zero for periodic and zero source terms
#          u = spysparselinalg.spsolve(A,B.dot(u))
#    # this operation is repeated many times.. you should
#    # prefactorize 'A' to speed up computation.
#       error = 0
#       for i in range(Nx):
#           error += ((u_ana[i] - u[i])**2)
#       rms_error = np.sqrt(error/Nx)
#       print(rms_error)
#       plot_x.append(Nx)
#       plot_y.append(rms_error)
# plt.plot(plot_x,plot_y)
# plt.title('RMS error vs. grid points for 2nd-order upwind scheme')
# plt.xlabel('Grid point')
# plt.ylabel('RMS error')
# plt.savefig('RMS error vs. grid points for 2nd-order upwind scheme.png')
# plt.show()

   # np.save('data_100_CN_1st',u)

#    if not(bool(np.mod(it,plot_every))): # plot every plot_every time steps
# #       ax.cla()
#        fig = plt.figure(0, figsize=(figwidth,figheight))
#        ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        ax.plot(x_mesh,u,'-k',linewidth=lineWidth)
#        ax.plot(x_mesh,u_initial(x_mesh),'--k',linewidth=1)
#        ax.plot(xx_ana,u_analytical(xx_ana),'r--',linewidth=1)
#        ax.text(0.7,0.9,r"$t="+"%1.5f" %time+"$",fontsize=gcafontSize,transform=ax.transAxes)
#        ax.grid('on',which='both')
#        plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#        plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#        ax.set_xlabel(r"$x$",fontsize=textFontSize)
#        ax.set_ylabel(r"$u(x,t)$",fontsize=textFontSize,rotation=90)
#        ax.set_ylim([ymin,ymax])
#
#        # Manually go frame by frame
#        # plt.draw()
#        # _ = input("Pausing \n")
#        # plt.show()
#        # plt.close(fig)
#
#        # Let it run as a movie
#        # plt.draw()
#        # plt.pause(0.1)
#        # plt.close(fig)
#
#
#        ### dereferecing figure object
#        fig=None
#
#        #plt.draw()
#        #       plt.tight_layout()
#        #       plt.tight_layout()
# #
# #       plt.cla()

# _ = input("Simulation Finished. Press Enter to continue...")
# plt.close()

##############
#P2.b)
# xx_10 = np.linspace(0,10,10)
# xx_25 = np.linspace(0,10,25)
# xx_50 = np.linspace(0,10,50)
# xx_100 = np.linspace(0,10,100)
# #For EE 1st Upwind
# u_10 = np.load('data_10_EE_1st.npy')
# u_25 = np.load('data_25_EE_1st.npy')
# u_50 = np.load('data_50_EE_1st.npy')
# u_100 = np.load('data_100_EE_1st.npy')
# u_ref = u_analytical(xx_100)
# plt.plot(xx_10,u_10,label='N=10')
# plt.plot(xx_25,u_25,label='N=25')
# plt.plot(xx_50,u_50,label='N=50')
# plt.plot(xx_100,u_100,label='N=100')
# plt.plot(xx_100,u_ref,'k--',label='Analytical solution')
# plt.legend()
# plt.title('Explicit Euler, 1st-order Upwind Scheme')
# plt.xlabel('Length')
# plt.ylabel('Value')
# plt.savefig('Explicit Euler, 1st-order Upwind Scheme.png')
# plt.show()
# #For EE 2st Central
# u_10 = np.load('data_10_EE_2nd_C.npy')
# u_25 = np.load('data_25_EE_2nd_C.npy')
# u_50 = np.load('data_50_EE_2nd_C.npy')
# u_100 = np.load('data_100_EE_2nd_C.npy')
# u_ref = u_analytical(xx_100)
# plt.plot(xx_10,u_10,label='N=10')
# plt.plot(xx_25,u_25,label='N=25')
# plt.plot(xx_50,u_50,label='N=50')
# plt.plot(xx_100,u_100,label='N=100')
# plt.plot(xx_100,u_ref,'k--',label='Analytical solution')
# plt.legend()
# plt.title('Explicit Euler, 2nd-order Central Scheme')
# plt.xlabel('Length')
# plt.ylabel('Value')
# plt.savefig('Explicit Euler, 2nd-order Central Scheme.png')
# plt.show()
# #For EE 2st Upwind
# u_10 = np.load('data_10_EE_2nd_U.npy')
# u_25 = np.load('data_25_EE_2nd_U.npy')
# u_50 = np.load('data_50_EE_2nd_U.npy')
# u_100 = np.load('data_100_EE_2nd_U.npy')
# u_ref = u_analytical(xx_100)
# plt.plot(xx_10,u_10,label='N=10')
# plt.plot(xx_25,u_25,label='N=25')
# plt.plot(xx_50,u_50,label='N=50')
# plt.plot(xx_100,u_100,label='N=100')
# plt.plot(xx_100,u_ref,'k--',label='Analytical solution')
# plt.legend()
# plt.title('Explicit Euler, 2nd-order Upwind Scheme')
# plt.xlabel('Length')
# plt.ylabel('Value')
# plt.savefig('Explicit Euler, 2nd-order Upwind Scheme.png')
# plt.show()
# #For CN 2st Upwind
# u_10 = np.load('data_10_CN_2nd_U.npy')
# u_25 = np.load('data_25_CN_2nd_U.npy')
# u_50 = np.load('data_50_CN_2nd_U.npy')
# u_100 = np.load('data_100_CN_2nd_U.npy')
# u_ref = u_analytical(xx_100)
# plt.plot(xx_10,u_10,label='N=10')
# plt.plot(xx_25,u_25,label='N=25')
# plt.plot(xx_50,u_50,label='N=50')
# plt.plot(xx_100,u_100,label='N=100')
# plt.plot(xx_100,u_ref,'k--',label='Analytical solution')
# plt.legend()
# plt.title('Crank Nicolson, 2nd-order Upwind Scheme')
# plt.xlabel('Length')
# plt.ylabel('Value')
# plt.savefig('Crank Nicolson, 2nd-order Upwind Scheme.png')
# plt.show()
# #For CN 2st Central
# u_10 = np.load('data_10_CN_2nd_C.npy')
# u_25 = np.load('data_25_CN_2nd_C.npy')
# u_50 = np.load('data_50_CN_2nd_C.npy')
# u_100 = np.load('data_100_CN_2nd_C.npy')
# u_ref = u_analytical(xx_100)
# plt.plot(xx_10,u_10,label='N=10')
# plt.plot(xx_25,u_25,label='N=25')
# plt.plot(xx_50,u_50,label='N=50')
# plt.plot(xx_100,u_100,label='N=100')
# plt.plot(xx_100,u_ref,'k--',label='Analytical solution')
# plt.legend()
# plt.title('Crank Nicolson, 2nd-order Central Scheme')
# plt.xlabel('Length')
# plt.ylabel('Value')
# plt.savefig('Crank Nicolson, 2nd-order Central Scheme.png')
# plt.show()
#For CN 1st Upwind
# u_10 = np.load('data_10_CN_1st.npy')
# u_25 = np.load('data_25_CN_1st.npy')
# u_50 = np.load('data_50_CN_1st.npy')
# u_100 = np.load('data_100_CN_1st.npy')
# u_ref = u_analytical(xx_100)
# plt.plot(xx_10,u_10,label='N=10')
# plt.plot(xx_25,u_25,label='N=25')
# plt.plot(xx_50,u_50,label='N=50')
# plt.plot(xx_100,u_100,label='N=100')
# plt.plot(xx_100,u_ref,'k--',label='Analytical solution')
# plt.legend()
# plt.title('Crank Nicolson, 1st-order Upwind Scheme')
# plt.xlabel('Length')
# plt.ylabel('Value')
# plt.savefig('Crank Nicolson, 1st-order Upwind Scheme.png')
# plt.show()
