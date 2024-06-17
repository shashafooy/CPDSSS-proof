from simulators.CPDSSS_models import CPDSSS
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from functools import reduce
from math import sqrt

def factors(n):
    """
    Find factors of given number.
    https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python

    Args:
        n (Int): Integer to find all factors of

    Returns:
        _type_: list of factors
    """
    step = 2 if n%2 else 1
    return list(set(reduce(list.__add__,([i,n//i] for i in range(1,int(sqrt(n)) + 1, step) if n % i == 0))))

N=20
L=5
M=int(N/L)
P=N-int(N/L)
max_T=6
T_range = range(N,N+max_T)


n_samples = 50000 #samples to generate per entropy calc

T=6


for N in range(40,41):
   
    plt.figure()             
    sim_model = CPDSSS(T,N,N)
    X,X_T,X_cond,G = sim_model.get_base_X_h(n_samples=n_samples)

    var=np.mean(np.var(X_T,axis=0))
    x=np.linspace(stats.norm.ppf(1e-6,scale=sqrt(var)),stats.norm.ppf(1-1e-6,scale=sqrt(var)),100)
    
    plt.plot(x,stats.norm.pdf(x,scale=sqrt(var)),lw=5)
    plt.hist(X_T,bins=100,density=True)
    plt.title("PDF N={0}".format(N))

plt.show()














