import numpy as np
from simulators.complex import mvn
class CPDSSS:
    def __init__(self, num_tx, N, L):
        self.T=num_tx
        self.N=N
        self.M=int(N/L)
        self.P=N-self.M
        self.sim_G = mvn(rho=0.0,dim_x=self.N*self.M)
        self.sim_Q = mvn(rho=0.0, dim_x=self.N*self.P)
        self.sim_S = mvn(rho=0.0, dim_x=self.M*self.T)
        self.sim_V = mvn(rho=0.0, dim_x=self.P*self.T)
        
        self.set_use_G_flag(g_flag=False)

    def sim(self, n_samples=1000):
        s = self.sim_S.sim(n_samples=n_samples).reshape((n_samples,self.M,self.T))
        v = self.sim_V.sim(n_samples=n_samples).reshape((n_samples,self.P,self.T))
        self.G = self.sim_G.sim(n_samples=n_samples).reshape((n_samples,self.N,self.M))
        Q = self.sim_Q.sim(n_samples=n_samples).reshape((n_samples,self.N,self.P))

        self.X=np.matmul(self.G,s) + np.matmul(Q,v)
        joint_X = self.X[:,:,0:self.T].reshape((n_samples,self.N*self.T),order='F')#order 'F' needed to make arrays stack instead of interlaced

        if(self.use_G):
            self.samples = np.concatenate((joint_X,self.G[:,:,0]),axis=1)
        else:
            self.samples = joint_X

        return self.samples
        # g_term = G[:,:,0]
        # xT_term = X[:,:,self.T-1]
        # xCond_term = X[:,:,0:self.T-1].reshape((n_samples,self.N*(self.T-1)),order='F')#order 'F' needed to make arrays stack instead of interlaced
    
    def set_use_G_flag(self,g_flag=True):
        self.use_G=g_flag
        if(g_flag):
            self.x_dim = self.N*self.T + self.N
        else:
            self.x_dim = self.N*self.T




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
 