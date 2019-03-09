#Simulation method for a 2-dim Levy process with infinity activity jumps using 
#shot noise method for series representation of infinitely divisible distributions
#using shot noise decomposition of the corresponding levy measure
#(refernece papers: Rosinski, Todorov-Tauchen)

import numpy as np
from math import exp
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import plot, show, grid, axis, xlabel, ylabel, title
#generates a one-dimensional gamma process
def my_gen_g():
    return np.random.gamma(1)
#generates a one-dimensional random process
def  my_gen_u():
     return np.random.rand(1)
# defines the function H(Gamma, V) for the represantation (5) in the paper Todorov
def  H(gamma, V, c, l):
     return V * (exp(-gamma / c) / l)
#defines a lambda function(throw away function) for short term...
def  my_H(c, l):
     return lambda gamma, V: H(gamma, V, c, l)
#generates a 2-dim exponential process
def  my_gen_v():
     return np.array([np.random.exponential(1),
            np.random.exponential(1)])
#generates one-dim poisson increments
def  my_gen_poisson_increments():
     return np.random.exponential(1)
#defines equidistant time observations
ts= np.linspace(0, 1, 1000)
#simulate 2-dim levy process
def  simulate_levy_w_jumps(ts, H, gen_poisson_increments, gen_v, gen_u, tau):
     dim = np.size(gen_v())
     levy = np.zeros((dim, np.size(ts)))
     gamma = 0
     while gamma < tau:
        gamma += gen_poisson_increments()

        V = gen_v()
        U = gen_u()

        vec = np.array([int(U <= t) for t in ts])
        levy += np.matmul(H(gamma, V).reshape((dim ,1)), vec.reshape((1, np.size(ts))))
     return levy
levy = simulate_levy_w_jumps(np.linspace(0, 1, 1000), my_H(1,1), my_gen_poisson_increments, my_gen_v, my_gen_u, 100)
#print(levy)
#plt.plot(np.linspace(0, 1, 1000), levy[0,:])
#plt.plot(np.linspace(0, 1, 1000), levy[1,:])
#title('2D Levy process')
#xlabel('t')
#ylabel('x_1, x_2')
#grid(True)
#plt.show()
