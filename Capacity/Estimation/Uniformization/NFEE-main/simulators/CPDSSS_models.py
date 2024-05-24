import numpy as np
import scipy.linalg as lin
from simulators.complex import mvn
import math 
class CPDSSS:
    def __init__(self, num_tx, N, L, use_gaussian_approx=True):
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
        self.set_use_h_flag(h_flag=False)
        self.sim_h_only = False

        self.fading = np.exp(-np.arange(self.N) / 3)

    def sim(self, n_samples=1000):
        """Generate samples X and (optional) G for CPDSSS

        Args:
            n_samples (int, optional): number of input samples. Defaults to 1000.            

        Returns:
            numpy: (n_samples, dim_x) array of generated values
        """
        
        s = self.sim_S.sim(n_samples=n_samples).reshape((n_samples,self.NL,self.T))
        v = self.sim_V.sim(n_samples=n_samples).reshape((n_samples,self.NNL,self.T))
        self.h = self.sim_H.sim(n_samples=n_samples) * np.sqrt(self.fading)
        if(self.gaussian_approx):
            g = self.sim_g.sim(n_samples=n_samples)
            self.G = np.zeros((n_samples,self.N,self.NL))
            #make toeplitz matrix
            for i in range(self.NL):
                self.G[:,:,i] = np.roll(g,shift=i*self.L,axis=1)

            self.G = self.sim_G.sim(n_samples=n_samples).reshape((n_samples,self.N,self.NL))
            Q = self.sim_Q.sim(n_samples=n_samples).reshape((n_samples,self.N,self.NNL))
        else:
            self.G,Q = self.sim_GQ(n_samples=n_samples)
        # import timeit
        # timeit.timeit(lambda: self.sim_GQ(n_samples=200000),number=1)

        self.X=np.matmul(self.G,s) + np.matmul(Q,v)
        joint_X = self.X[:,:,0:self.T].reshape((n_samples,self.N*self.T),order='F')#order 'F' needed to make arrays stack instead of interlaced

        if(self.use_h):
            self.samples = np.concatenate((joint_X,self.h),axis=1)
        else:
            self.samples = joint_X
        if(self.sim_h_only):
            self.samples = self.h

        return self.samples
        # g_term = G[:,:,0]
        # xT_term = X[:,:,self.T-1]
        # xCond_term = X[:,:,0:self.T-1].reshape((n_samples,self.N*(self.T-1)),order='F')#order 'F' needed to make arrays stack instead of interlaced
    
    def sim_GQ(self,n_samples=1000):
        # z=np.exp(1j*2*np.pi*np.arange(0,self.N)**2 / self.N)/np.sqrt(self.N)
        # Z=lin.toeplitz(z,np.concatenate(([z[0]], z[-1:0:-1])))

        # E=np.eye(self.N)
        # E=E[:,0::self.L]

        # a=np.zeros((self.NL,1))
        # a[0]=1

        #Flat fading
        
        G = np.zeros((n_samples,self.N,self.NL))
        Q = np.zeros((n_samples,self.N,self.NNL))
        # from datetime import timedelta
        # import time

        # start_time = time.time()
        for i in range(n_samples):
            # H = lin.toeplitz(h[i,:],np.concatenate(([h[i,0]],h[i,-1:0:-1])))
            # A=E.T @ H 
            #Slightly faster than doing E.T @ H
            HE = lin.toeplitz(self.h[i,:],np.concatenate(([self.h[i,0]],self.h[i,-1:0:-1])))[0::self.L,:]
            R=HE.T @ HE + 0.0001*np.eye(self.N)                       
            # p=A.T @ a
            p = HE[0,:].T
            g=lin.inv(R)@p

            #Only take every L columns of toepltiz matrix
            # Slightly faster than doing G@E
            G[i,:,:]=lin.toeplitz(g,np.concatenate(([g[0]], g[-1:0:-1])))[:,0::self.L]
            
            ev,V = lin.eig(R)
            #Only use eigenvectors associated with "zero" eigenvalue
            #  get indices of the smallest eigenvalues to find "zero" EV
            sort_indices = np.argsort(ev)
            Q[i,:,:]=V[:,sort_indices[0:self.NNL]]
        # end_time = time.time()
        # print("GQ time: ",str(timedelta(seconds = int(end_time - start_time))))       

        return G,Q


    def chan_entropy(self):
        return 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.diag(self.fading)))




    def set_use_h_flag(self,h_flag=True):
        """Enables sim to append G in addition to X

        Args:
            g_flag (bool, optional): Enable G output. Defaults to True.
        """
        self.use_h=h_flag
        if(h_flag):
            self.x_dim = self.N*self.T + self.N
        else:
            self.x_dim = self.N*self.T

    def set_sim_h_only(self,sim_h=True):
        """Set flag to only return G when using sim()
        
        Args:
            sim_g (bool): True if you want sim() to return only G
        """
        self.sim_h_only = sim_h
        if(sim_h):
            self.x_dim = self.N
        elif(self.use_h):
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
        self.set_use_h_flag(h_flag=True)
        vals = self.sim(n_samples=n_samples)
        X=vals[:,0:self.N*self.T]
        X_T=X[:,-self.N:]
        X_cond = X[:,0:-self.N]
        return X,X_T,X_cond,self.h


class CPDSSS_X:
    """
    Generate the output samples X for CPDSSS
    """
    def __init__(self, num_tx, N, L):
        self.base_model=CPDSSS(num_tx=num_tx,N=N,L=L)

    def sim(self, n_samples=1000):
        self.base_model.gen_XG(n_samples=n_samples)
        T=self.base_model.T
        N=self.base_model.N
        #order 'F' needed to make arrays stack instead of interlaced
        return self.base_model.X[:,:,0:T].reshape((n_samples,N*T),order='F')
    
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
 