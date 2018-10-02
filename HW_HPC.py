import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

##(a)
#N_procs = 1
def fx(x):
    return np.sin(x)
def fpx(x):
    return np.cos(x)
plot_x = np.zeros(99999999)
plot_y = np.zeros(99999999)
for index in range(99999999, 0, -1):
    delta_x = index/1000000000
    n = int(1/delta_x)
    a = [[0 for row in range(n)] for col in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0:
                a[i][j] = 0
            elif i == n:
                a[i][j] = 0
            elif j == i:
                a[i][j] =  - 1./(2.*delta_x)
            elif j ==i+1:
                a[i][j] = 0
            elif j ==i+2:
                a[i][j] =  1./(2.*delta_x)
            else:
                a[i][j] = 0.

    b = [[0] for col in range(n)]
    for i in range(n):
        b[i][0] = fx(delta_x*i)
    c = [[0] for col in range(n)]
    for i in range(n):
        if i == 0:
            c[i][0] = 1
        elif i == n:
            c[i][0] = fpx(1)
        else:
            c[i][0] = 0
    npb = np.array(b)
    npa = np.array(a)
    npc = np.array(c)
    res = np.dot(npa, npb) + npc
    abssum = 0
    func = [[0] for col in range(n)]
    for i in range(n):
        abssum += ((fpx(i*delta_x) - res[i])**2)
    plot_y[index - 1] = np.sqrt(abssum/n)
    plot_x[index-1] = 1/delta_x
plt.loglog(plot_x,plot_y)
# plt.xlabel('Inverse of the grid spacing')
# plt.ylabel('RMS error')
# plt.title('2nd order accuracy scheme')
# plt.savefig('2nd order accuracy scheme.png')
plt.show()