import numpy as np
import scipy.linalg as lin
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
        self.sim_H = mvn(rho=0.0, dim_x=self.N)
        
        self.set_use_G_flag(g_flag=False)
        self.sim_g_only = False

    def sim(self, n_samples=1000):
        """Generate samples X and (optional) G for CPDSSS

        Args:
            n_samples (int, optional): number of input samples. Defaults to 1000.            

        Returns:
            numpy: (n_samples, dim_x) array of generated values
        """
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
        if(self.sim_g_only):
            self.samples = self.G[:,:,0]


        return self.samples
        # g_term = G[:,:,0]
        # xT_term = X[:,:,self.T-1]
        # xCond_term = X[:,:,0:self.T-1].reshape((n_samples,self.N*(self.T-1)),order='F')#order 'F' needed to make arrays stack instead of interlaced
    
    def set_use_G_flag(self,g_flag=True):
        """Enables sim to append G in addition to X

        Args:
            g_flag (bool, optional): Enable G output. Defaults to True.
        """
        self.use_G=g_flag
        if(g_flag):
            self.x_dim = self.N*self.T + self.N
        else:
            self.x_dim = self.N*self.T

    def set_sim_G_only(self,sim_g=True):
        """Set flag to only return G when using sim()
        
        Args:
            sim_g (bool): True if you want sim() to return only G
        """
        self.sim_g_only = sim_g
        if(sim_g):
            self.x_dim = self.N
        elif(self.use_G):
            self.x_dim = self.N*self.T + self.N
        else:
            self.x_dim = self.N*self.T

    
    def get_base_X_G(self,n_samples=1000):
        """Generate values for G and X

        Args:
            n_samples (int, optional): Number of input samples. Defaults to 1000.

        Returns:
            numpy: X, X_T, X_cond, G
        """
        self.set_use_G_flag(g_flag=True)
        vals = self.sim(n_samples=n_samples)
        X=vals[:,0:self.N*self.T]
        X_T=X[:,-self.N:]
        X_cond = X[:,0:-self.N]
        G = vals[:,-self.N:]
        return X,X_T,X_cond,G

    def generate_G_Q(N,L,n_samples):
        epsilon=1e-4
        h=self.sim_H.sim(n_samples=n_samples) + 1j*np.self.sim_H.sim(n_samples=n_samples)
        h=h*np.exp(-np.array(range(0,N))/3)
        # H=lin.toeplitz(h,r=)

'''
function [G,Q] = generate_G_Q(N,L)
epsilon=1e-4;

h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
h=randn(N,1).*exp(-[0:N-1]'/3); % REAL EXPERIMENT
H=toeplitz(h,[h(1); h(end:-1:2)]);
E=eye(N/L);E=upsample(E,L);

A=E'*H;
R=A'*A+epsilon*eye(N);
a=[1; zeros(N/L-1,1)];
p=A'*a;
g=R\p;

G=toeplitz(g,[g(1); g(end:-1:2)]);
G=G*E;

[V D]=eig(R);
NNL=N-N/L;
P=NNL;
Q=V(:,1:NNL);
% Q=Q';       %take hermitian to match X=SG + VQ
% v=(randn(T,P)+1i*randn(T,P))/sqrt(2);


M=N/L;P=N-M;
G=randn(N,M);
Q=randn(N,P);
end
'''

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
 