import abc
import numpy as np
import scipy.linalg as lin
import scipy.special as spec
import scipy.stats as stats
from simulators.complex import mvn
import math 
from joblib import Parallel,delayed
import theano.tensor as tt


class _distribution:
    __metaclass__ = abc.ABCMeta

    def __init__(self,x_dim):
        self.x_dim=x_dim

    @abc.abstractmethod
    def sim(self):
        """Method to generate sample from the distribution"""
        return
        
    def entropy(self):
        """Method returning the entropy of the given distribution. Use default method if entropy unknown.
          Defaults to returning none."""
        return None
    def pdf(self,x):
        """PDF of the distribution"""
        return None
    def logpdf(self,x):
        """log(pdf) of the distribution. Useful for entropy calculations"""
        return None
    
class CPDSSS(_distribution):
    def __init__(self, num_tx, N, L, use_gaussian_approx=False):
        super().__init__(x_dim=N*num_tx+N)
        self.T=num_tx
        self.N=N
        self.L=L
        self.NL=int(N/L)
        self.NNL=N-self.NL
        self.sim_G = mvn(rho=0.0,dim_x=self.N*self.NL)
        self.sim_g = mvn(rho=0.0,dim_x=self.N)
        self.sim_Q = mvn(rho=0.0, dim_x=self.N*self.NNL)
        self.sim_S = mvn(rho=0.0, dim_x=self.NL*self.T)
        self.sim_V = mvn(rho=0.0, dim_x=self.NNL*self.T)
        self.sim_H = mvn(rho=0.0, dim_x=self.N)

        self.gaussian_approx = use_gaussian_approx        
        self.use_chan_in_sim()
        self._sim_chan_only = False

        self.fading = np.exp(-np.arange(self.N) / 3)

    def sim(self, n_samples=1000):
        """Generate samples X and (optional) G for CPDSSS

        Args:
            n_samples (int, optional): number of input samples. Defaults to 1000.            

        Returns:
            numpy: (n_samples, dim_x) array of generated values
        """
        
        self.s = self.sim_S.sim(n_samples=n_samples).reshape((n_samples,self.NL,self.T))
        v = self.sim_V.sim(n_samples=n_samples).reshape((n_samples,self.NNL,self.T))
        self.h = self.sim_H.sim(n_samples=n_samples) * np.sqrt(self.fading)

        if(self._sim_chan_only): #return early if we only need channel
            self.samples = self.h
            return self.h

        if(self.gaussian_approx): #use approximation that G,Q are gaussian
            g = self.sim_g.sim(n_samples=n_samples)
            self.G = np.zeros((n_samples,self.N,self.NL))
            #make toeplitz matrix
            for i in range(self.NL):
                self.G[:,:,i] = np.roll(g,shift=i*self.L,axis=1)

            self.G = self.sim_G.sim(n_samples=n_samples).reshape((n_samples,self.N,self.NL))
            Q = self.sim_Q.sim(n_samples=n_samples).reshape((n_samples,self.N,self.NNL))
        else: #Evaluate true G,Q according to CPDSSS
            self.G,Q = self.sim_GQ(n_samples=n_samples)
        # import timeit
        # timeit.timeit(lambda: self.sim_GQ(n_samples=200000),number=1)

        self.X=np.matmul(self.G,self.s) + np.matmul(Q,v)
        joint_X = self.X[:,:,0:self.T].reshape((n_samples,self.N*self.T),order='F')#order 'F' needed to make arrays stack instead of interlaced

        if(self._use_chan):
            self.samples = np.concatenate((joint_X,self.h),axis=1)
        else:
            self.samples = joint_X        

        return self.samples
        # g_term = G[:,:,0]
        # xT_term = X[:,:,self.T-1]
        # xCond_term = X[:,:,0:self.T-1].reshape((n_samples,self.N*(self.T-1)),order='F')#order 'F' needed to make arrays stack instead of interlaced
    
    def sim_GQ(self,n_samples=1000):        
        import multiprocessing
        # start_time = time.time()
        pool = multiprocessing.Pool()
        results = pool.map(self._gen_GQ_sample,[self.h[i,:] for i in range(n_samples)])

        pool.close()
        pool.join()
        # print(f"pool time: {time.time() - start_time}")

        G_results,Q_results = zip(*results)
        G=np.array(G_results)
        Q=np.array(Q_results)

        
        return G,Q
    
    def _gen_GQ_sample(self,h):
        # H = lin.toeplitz(h[i,:],np.concatenate(([h[i,0]],h[i,-1:0:-1])))
            # A=E.T @ H 
            #Slightly faster than doing E.T @ H
        HE = lin.toeplitz(h,np.concatenate(([h[0]],h[-1:0:-1])))[0::self.L,:]
        R=HE.T @ HE + 0.0001*np.eye(self.N)                       
        p = HE[0,:].T
        g=lin.inv(R)@p
        #Only take every L columns of toepltiz matrix
        # Slightly faster than doing G@E
        G=lin.toeplitz(g,np.concatenate(([g[0]], g[-1:0:-1])))[:,0::self.L]
        
        ev,V = lin.eig(R)
        #Only use eigenvectors associated with "zero" eigenvalue
        #  get indices of the smallest eigenvalues to find "zero" EV
        sort_indices = np.argsort(ev)
        #Sometimes get a very small imaginary value, ignore it
        Q=np.real(V[:,sort_indices[0:self.NNL]])

        return G,Q

    
    def chan_entropy(self):
        return 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.diag(self.fading)))


    def use_chan_in_sim(self,h_flag=True):
        """Enables sim to append G in addition to X

        Args:
            g_flag (bool, optional): Enable G output. Defaults to True.
        """
        self._use_chan=h_flag
        if(h_flag):
            self.x_dim = self.N*self.T + self.N
        else:
            self.x_dim = self.N*self.T

    def only_sim_chan(self,sim_h=True):
        """Set flag to only return G when using sim()
        
        Args:
            sim_g (bool): True if you want sim() to return only G
        """
        self._sim_chan_only = sim_h
        if(sim_h):
            self.x_dim = self.N
        elif(self._use_chan):
            self.x_dim = self.N*self.T + self.N
        else:
            self.x_dim = self.N*self.T

    
    def get_base_X_h(self,n_samples=1000):
        """Generate values for G and X

        Args:
            n_samples (int, optional): Number of input samples. Defaults to 1000.

        Returns:
            numpy: X, X_T, X_cond, G
        """
        self.use_chan_in_sim(h_flag=True)
        vals = self.sim(n_samples=n_samples)
        X=vals[:,0:self.N*self.T]
        X_T=X[:,-self.N:]
        X_cond = X[:,0:-self.N]
        return X,X_T,X_cond,self.h
    
    def set_dim_hxc(self):
        self.x_dim = self.N*(self.T-1) + self.N

    def set_dim_xxc(self):
        self.x_dim = self.N*self.T

    def set_dim_cond(self):
        self.x_dim = self.N*(self.T-1)

    def set_dim_joint(self):
        self.x_dim = self.N*self.T + self.N


class CPDSSS_XS(CPDSSS):
    """
    Generate the output samples X for CPDSSS
    """
    def __init__(self, N, L):
        CPDSSS.__init__(self,num_tx=1,N=N,L=L,use_gaussian_approx=False)
        # self.base_model=CPDSSS(num_tx=num_tx,N=N,L=L,use_gaussian_approx=False)

    def sim_use_X(self):
        self.use_X=True
        self.use_S=False
        self.x_dim=self.N
    
    def sim_use_S(self):
        self.use_X=True
        self.use_S=False
        self.x_dim=self.NL
    
    def sim_use_XS(self):
        self.use_X=True
        self.use_S=True
        self.x_dim=self.N+self.NL


    def sim(self, n_samples=1000):
        X = super().sim(n_samples=n_samples)
        ret_val = np.empty((n_samples,0))
        if self.use_X:
            ret_val = np.concatenate((ret_val,X),axis=1)
        if self.use_S:
            ret_val = np.concatenate((ret_val,self.s[:,:,0]),axis=1)

        return ret_val
    
class CPDSSS_XG:
    """
    Used to generate the joint samples between X and G for CPDSSS
    """
    def __init__(self, num_tx, N, L):
        self.base_model=CPDSSS(num_tx=num_tx,N=N,L=L)

    def sim(self, n_samples=1000):
        self.base_model.gen_XG(n_samples=n_samples)
        T=self.base_model.T
        N=self.base_model.N
        #order 'F' needed to make arrays stack instead of interlaced
        joint_X=self.base_model.X[:,:,0:T].reshape((n_samples,N*T),order='F')
        g_term = self.base_model.G[:,:,0]
        return np.concatenate((joint_X,g_term),axis=1)
 
class Exponential(_distribution):
    """Class to simulate an exponential distribution
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    def __init__(self,lamb):
        super.__init__(x_dim=1)
        self.lamb=lamb
    
    def sim(self,n_samples):        
        return np.random.default_rng().exponential(scale = 1/self.lamb, size = (n_samples,1))
    
    def entropy(self):
        return 1- np.log(self.lamb)
    
class Exponential_sum(_distribution):
    """Class to simulate the sum of two independent exponential variables
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    def __init__(self,lambda1,lambda2):
        super().__init__(x_dim=1)
        assert lambda1>lambda2, "lambda1 > lambda2 for this entropy test"
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.expo1=Exponential(lambda1)
        self.expo2=Exponential(lambda2)

    def sim(self,n_samples):
        return self.expo1.sim(n_samples) + self.expo2.sim(n_samples)
    
    def entropy(self):
        return 1 + np.euler_gamma + np.log((self.lambda1-self.lambda2)/(self.lambda1*self.lambda2)) + spec.digamma(self.lambda1/(self.lambda1 - self.lambda2))
    
class Laplace(_distribution):
    """Simulate the laplace distribution
    https://en.wikipedia.org/wiki/Laplace_distribution"""
    def __init__(self,mu,b,N=1):        
        super().__init__(x_dim=N)
        self.mu=mu
        self.b=b
    
    def sim(self,n_samples):
        return np.random.default_rng().laplace(self.mu,self.b,(n_samples,self.x_dim))
    
    def entropy(self):
        return (1 + np.log(2*self.b))*self.x_dim
    def pdf(self,x):
        return 1/(2*self.b)*np.exp(-np.abs(x-self.mu)/self.b)
    def logpdf(self,x):
        return -self.x_dim*np.log(2*self.b) - tt.sum(tt.abs_(x-self.mu)/self.b,axis=1)
    
class Cauchy(_distribution):
    def __init__(self,mu,gamma,N=1):        
        super().__init__(x_dim=N)
        self.mu=mu
        self.gamma=gamma

    def sim(self,n_samples):
        return stats.cauchy.rvs(loc=self.mu,scale=self.gamma, size=(n_samples,self.x_dim))
    
    def entropy(self):
        return np.log(4*np.pi*self.gamma)*self.x_dim
    
class Logistic(_distribution):
    def __init__(self,mu,s,N=1):
        super().__init__(x_dim=N)
        self.s=s
        self.mu=mu

    def sim(self,n_samples):
        return stats.logistic.rvs(loc=self.mu,scale=self.s,size=(n_samples,self.x_dim))
    
    def entropy(self):
        return (np.log(self.s) + 2)*self.x_dim

