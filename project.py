import numpy as np
import matplotlib.pyplot as plt

L = 1 # Length of the square
Re = 1000 # Reynolds number
N = 512 # Number of grids in one direction
T = 200 # End time
dt = 0.0005 # Step size in time
ds = L / N # Step size in length
Nt = T / dt # Number of time steps
N_plot = np.linspace(0,N-1,N)


## Initial/Boundary conditions
# Left
ul = np.zeros(N)
vl = np.zeros(N)
sl = np.zeros(N) # Boundary value of streamfunction
# Right
ur = np.zeros(N)
vr = np.zeros(N)
sr = np.zeros(N)
# Bottom
ub = np.zeros(N)
vb = np.zeros(N)
sb = np.zeros(N)
# Top
ut = np.ones(N)
vt = np.zeros(N)
st = np.zeros(N)


u0 = np.zeros((N,N))
v0 = np.zeros((N,N))
w0 = np.zeros((N,N))
psi0 = np.zeros((N,N)) # Initial sreamfunction
wleft = np.zeros(N) # Boundary value of vorticity
wright = np.zeros(N)
wbottom = np.zeros(N)
wtop = np.zeros(N)

# Initializing vorticity field
w = np.zeros((N,N))
f = np.zeros((N,N)) # Increment of vorticity based on time
w[:,:-1] -= v0[:,1:]
w[:,1:] += v0[:,:-1]
w[:-1,:] += u0[1:,:]
w[1:,:] -= u0[:-1,:]
w[:,0] += vl
w[:,-1] -= vr
w[0,:] -= ub
w[-1,:] += ut
w /= (2 * ds)

# Initializing streamfunction
psi = np.zeros((N,N))
res = np.zeros(psi.shape)
res_1 = np.zeros(psi.shape)
res_rms = 1
iter = 0
while res_rms > 0.0001:
     psi[:-1,:] += psi0[1:,:]
     psi[1:,:] += psi0[:-1,:]
     psi[:,:-1] += psi0[:,1:]
     psi[:,1:] += psi0[:,:-1]
     psi[:,0] += sl
     psi[0,:] += sb
     psi[:,-1] += sr
     psi[-1,:] += st
     psi += w * (ds ** 2)
     psi *= 0.25

     res[:-1,:] -= psi[1:,:]
     res[1:,:] -= psi[:-1,:]
     res[:,:-1] -= psi[:,1:]
     res[:,1:] -= psi[:,:-1]
     res[:,0] -= sl
     res[0,:] -= sb
     res[:,-1] -= sr
     res[-1,:] -= st
     res -= w * (ds ** 2)
     res *= 0.25
     res += psi
     res_rms = np.sqrt(np.mean(res ** 2))
     psi0 = psi
     iter += 1

# Solving
up = np.zeros((N,N))
um = np.zeros(up.shape)
vp = np.zeros(up.shape)
vm = np.zeros(up.shape)

time_step = 0
while time_step < Nt:
     # Boundary conditions for vorticity
     wleft = -2 * vl / ds + 2 * (sl - psi[:,0]) / ds ** 2
     wright = 2 * vr / ds + 2 * (sr - psi[:,-1]) / ds ** 2
     wbottom = 2 * ub / ds + 2 * (sb - psi[0,:]) / ds ** 2
     wtop = -2 * ut / ds + 2 * (st - psi[-1,:]) / ds ** 2

     # Solving vorticity fields
     up_term = np.zeros(up.shape)
     up_term[:-1,:] += psi[1:,:]
     up_term[1:,:] -= psi[:-1,:]
     up_term[0,:] -= sb
     up_term[-1,:] += st
     up = 0.25 * (up_term + abs(up_term)) / ds

     um_term = np.zeros(up.shape)
     um_term[:-1,:] += psi[1:,:]
     um_term[1:,:] -= psi[:-1,:]
     um_term[0,:] -= sb
     um_term[-1,:] += st
     um = 0.25 * (um_term - abs(um_term)) / ds

     vp_term = np.zeros(vp.shape)
     vp_term[:,:-1] -= psi[:,1:]
     vp_term[:,1:] += psi[:,:-1]
     vp_term[:,0] += sl
     vp_term[:,-1] -= sr
     vp = 0.25 * (vp_term + abs(-vp_term)) / ds

     vm_term = np.zeros(vp.shape)
     vm_term[:,:-1] -= psi[:,1:]
     vm_term[:,1:] += psi[:,:-1]
     vm_term[:,0] += sl
     vm_term[:,-1] -= sr
     vm = 0.25 * (vm_term - abs(-vm_term)) / ds

     # Discretization of f
     # f = (w[1:,:]+w[:-1,:]-4*w+w[:,1:]+w[:,:-1])/(Re*ds**2)
     # f -= ((psi[1:,:]-psi[:-1,:])*(w[:,1:]-w[:,:-1])+(psi[:,:-1]-psi[:,1:])*(w[1:,:]-w[:-1,:]))/(4*ds**2)
     # f += 0.5*(up+um)*(w[:,1:]+w[:,:-1]-2*w)/ds
     # f += 0.5*(vp+vm)*(w[1:,:]+w[:-1,:]-2*w)/ds

     f = np.zeros(up.shape)
     row1 = np.zeros(up.shape)
     row1[:-1, :] += w[1:,:]
     row1[1:, :] += w[:-1,:]
     row1[:, :-1] += w[:,1:]
     row1[:, 1:] += w[:,:-1]
     row1[0,:] += wbottom
     row1[-1,:] += wtop
     row1[:,0] += wleft
     row1[:,-1] += wright
     row1 -= 4 * w
     f += row1 / (Re * ds ** 2)

     row2 = np.zeros(up.shape)
     term1 = np.zeros(up.shape)
     term1[:-1,:] += psi[1:,:]
     term1[1:,:] -= psi[:-1,:]
     term1[0, :] -= sb
     term1[-1, :] += st
     term2 = np.zeros(up.shape)
     term2[:,:-1] += w[:,1:]
     term2[:,1:] -= w[:,:-1]
     term2[:, 0] -= wleft
     term2[:, -1] += wright
     term3 = np.zeros(up.shape)
     term3[:,1:] += psi[:,:-1]
     term3[:,:-1] -= psi[:,1:]
     term3[:, 0] += sl
     term3[:, -1] -= sr
     term4 = np.zeros(up.shape)
     term4[:-1,:] += w[1:,:]
     term4[1:,:] -= w[:-1,:]
     term4[0, :] -= wbottom
     term4[-1, :] += wtop
     row2 += (term1 * term2) + (term3 * term4)
     f -= row2 / (4 * ds ** 2)

     row3 = np.zeros(up.shape)
     term5 = up + um
     term6 = np.zeros(up.shape)
     term6[:,:-1] += w[:,1:]
     term6[:,1:] += w[:,:-1]
     term6[:, 0] += wleft
     term6[:, -1] += wright
     term6 -= 2 * w
     row3 = 0.5 * term5 * term6 / ds
     f += row3

     row4 = np.zeros(up.shape)
     term7 = vp + vm
     term8 = np.zeros(up.shape)
     term8[:-1,:] += w[1:,:]
     term8[1:,:] += w[:-1,:]
     term8[0, :] += wbottom
     term8[-1, :] += wtop
     term8 -= 2 * w
     f += 0.5 * term7 * term8 / ds
     w += dt * f

     # Update streamfunction based on the new vorticity
     res_rms_1 = 1
     psi = np.zeros((N,N))
     iter = 0
     while res_rms_1 > 0.0001:
          psi[:-1, :] += psi0[1:, :]
          psi[1:, :] += psi0[:-1, :]
          psi[:, :-1] += psi0[:, 1:]
          psi[:, 1:] += psi0[:, :-1]
          psi[:, 0] += sl
          psi[0, :] += sb
          psi[:, -1] += sr
          psi[-1, :] += st
          psi += w * (ds ** 2)
          psi *= 0.25

          res_1[:-1, :] -= psi[1:, :]
          res_1[1:, :] -= psi[:-1, :]
          res_1[:, :-1] -= psi[:, 1:]
          res_1[:, 1:] -= psi[:, :-1]
          res_1[:, 0] -= sl
          res_1[0, :] -= sb
          res_1[:, -1] -= sr
          res_1[-1, :] -= st
          res_1 -= w * (ds ** 2)
          res_1 *= 0.25
          res_1 += psi
          res_rms_1 = np.sqrt(np.mean(res ** 2))
          psi0 = psi
          iter += 1

     if time_step % 2000 == 0:
         print(time_step / 2000)
     time_step += 1


plt.contour(N_plot,N_plot,-w,300,colors='k')
plt.clabel(plt.contour(N_plot,N_plot,-w,300,colors='k'), fontsize=9, inline=1)
plt.title('Vorticity contours for Re = 1000')
plt.savefig('vort_1000.png')
plt.show()
plt.contour(N_plot,N_plot,psi0,15,colors='k')
plt.clabel(plt.contour(N_plot,N_plot,psi0,15,colors='k'), fontsize=9, inline=1)
plt.title('Streamline pattern for Re = 1000')
plt.savefig('str_1000.png')
plt.show()

# Find velocity field
u = np.zeros((N,N))
v = np.zeros(u.shape)
u[:-1,:] += psi[1:,:] / ds
u[1:,:] -= psi[:-1,:] / ds
u[0,:] -= sb / ds
u[-1,:] += st / ds
v[:,:-1] -= psi[:,1:] / ds
v[:,1:] += psi[:,:-1] / ds
v[:,-1] -= sr / ds
v[:,0] += sl / ds


# Velocity contour, not shown in the report
plt.quiver(N_plot,N_plot,u,v)
plt.title('Velocity field for Re = 1000')
plt.savefig('vel_1000.png')
plt.show()


print(-w[-1,163])
print(-w[-1,192])
print(-w[-1,224])
print(-w[-1,256])
print(-w[-1,288])
print(-w[-1,320])
print(-w[-1,352])

print(psi0[-1,163])
print(psi0[-1,192])
print(psi0[-1,224])
print(psi0[-1,256])
print(psi0[-1,288])
print(psi0[-1,320])
print(psi0[-1,352])
#0.3125 - 0.6875
# Re = 100 vortcity
# 9.954279133751125
# 8.385881080540262
# 7.21606326399029
# 6.398036425950681
# 5.940800146391011
# 5.891020368828238
# 6.329594847099954
# Str_fcn
# -0.007455242610794444
# -0.007504096780873392
# -0.00753985172803647
# -0.007563805888900341
# -0.007575466041578658
# -0.007573149183111929
# -0.007554100777717222

# Re = 1000 vorticity
# 24.015738037710662
# 19.5173741458666
# 16.117659234171416
# 14.158521537111158
# 13.33138449775373
# 13.349821078924672
# 13.997839523881483
#str_fcn
# -0.001908393090879789
# -0.0019170713860973167
# -0.0019233309085336123
# -0.001926653669423456
# -0.0019277468173898255
# -0.001927222544043555
# -0.0019255165272969086






