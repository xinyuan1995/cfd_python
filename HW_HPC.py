import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import math

##(a)

def fx(x):
    return np.sin(x)
def fpx(x):
    return np.cos(x)

FROM_INDEX = 1 # FROM 10^1
TO_INDEX = 4 # TO 10^4
DELTA_INDEX = 0.5 # WITH MULTIPLIER 10^0.5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()#The index of the thread
size = comm.Get_size()#The total number of all threads

if rank == 0:
    # DO master
    index = FROM_INDEX
    plot_x = []
    plot_y = []
    plot_z = []
    wt = []
    while index <= TO_INDEX:
        N = round(10 ** index)#Current N
        delta_x = 1. / N
        res = 0.0
        #send to slave node
        for process in range(1, size):
            comm.send([process, N], dest=process, tag=1)#Send task to slave node

        received_processes = 0
        s = 0.0
        while received_processes < size - 1:#If not all nodes collected, do following
            s += comm.recv(source=MPI.ANY_SOURCE, tag=1) #Receive result from slave node
            process = comm.recv(source=MPI.ANY_SOURCE, tag=2) #Receive counts from slave node
            received_processes += 1

        # average on each res
        s /= (size - 1) #Do average over each result
        print(N, s)#Result for N Node
        index += DELTA_INDEX

        #Append new point to plot arrays
        plot_y.append(np.sqrt(s/N))
        plot_x.append(1/delta_x)
        plot_z.append(delta_x)
        wt.append(MPI.Wtime())

    #Shut down all slave nodes
    for process in range(1,size):
        comm.send([-1, -1], dest=process, tag=1)

    #Plot
    plt.loglog(plot_x,plot_y)
    plt.xlabel('Inverse of the grid spacing')
    plt.ylabel('RMS error')
    plt.title('N_procs = 1')
    plt.loglog(plot_x,plot_z,'r--',label='n = 1')
    plt.legend()
    plt.savefig('N_procs = 1.png')
    plt.show()
    print(wt[-1]-wt[0])
#Time for N_procs = 1 is 12.376559019088745s
#Time for N_procs = 2 is 
#Time for N_procs = 4 is 
#Time for N_procs = 8 is 
#Time for N_procs = 16 is 
else:
    #DO slave
    while True:
        process, N = comm.recv(source=0, tag=1)
        if process == -1:
            break

        #The start node index and the end node index for this part of matrix
        start_node = int((process - 1) * N)
        end_node = math.ceil(process * N) - 1
        n = end_node - start_node + 3 #Size of the matrix a (n*n) and vector b (n * 1)

        #Calculate u_prime
        a = [[0]*n for row in range(n)]
        b = [0] * n
        delta_x = 1. / N
        a[0][0] = -1. / delta_x
        a[0][1] = +1. / delta_x
        for i in range(1, n-1):
            a[i][i-1] = -.5 / delta_x
            a[i][i+1] = +.5 / delta_x
        a[n-1][n-2] = -1. / delta_x
        a[n-1][n-1] = +1. / delta_x
        for i in range(n):
            b[i] = fx(delta_x * (i + start_node - 1))#Part of vector b
        res = np.dot(a,b)
        abssum = 0.0
        abssum = 0
        for i in range(n):
            abssum += ((fpx(delta_x * (i + start_node - 1)) - res[i]) ** 2)
        abssum = np.sqrt(abssum / n)

        #send results and task number back to master
        comm.send(abssum, dest=0, tag=1)
        comm.send(rank, dest=0, tag=2)