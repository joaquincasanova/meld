import numpy as np
import scipy.special as sp
from numpy import matlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class mind:
    def __init__(self,p):
        self.I=np.eye(p)#[]
        self.r=0.56#[1/mV]
        #Generate new values for DCM params. See David et al 2006.
        self.He=np.exp(np.random.randn(p,1)*1./16.+np.log(4.0))*1e-3#[s]
        self.Hi=32.0*np.ones([p,1])*1e-3#[s]
        self.tau_e=np.exp(np.random.randn(p,1)*1./16.+np.log(4.0))*1e-3#[s]
        self.tau_i=16.0*np.ones([p,1])*1e-3#[s]
        self.delta=np.exp(np.random.randn(p,p)*1./16.+np.log(16.0))*1e-3#[s]
        for ii in range(0,p):
            self.delta[ii,ii]=2.*1e-3#[s]
        self.gamma_1=1.0#[mV]
        self.gamma_2=0.8#[mV]
        self.gamma_3=0.25#[mV]
        self.gamma_4=0.25#[mV]
        self.eta_1=np.exp(np.random.randn(1,1)*1./16.+np.log(96.0))
        self.eta_2=np.exp(np.random.randn(1,1)*1./16.+np.log(1024.0))
        self.thetaC=np.random.randn(p,1)
        self.cU=np.exp(np.random.randn(p,1)*1./2.+0.0)#[mV]
        self.cB=np.exp(np.random.randn(p,p)*1./2.+np.log(16.0))#[mV]
        self.cL=np.exp(np.random.randn(p,p)*1./2.+np.log(4.0))#[mV]
        self.cF=np.exp(np.random.randn(p,p)*1./2.+np.log(32.0))#[mV]

    def S(self,x):
        s=1.0/(1.0+np.exp(-self.r*x))-0.5
        return s

    def gdf(self,t,del_t):
        #self.eta_2**self.eta_1*t**(self.eta_1-1.)*np.exp(-self.eta_2*t)/sp.gamma(self.eta_1)
        if t==0:            
            return 1.0/del_t#[1/s]
        else:
            return 0.0
    def u(self,t,p,q_init,batch_size,del_t):
        pp=np.argmax(q_init,0)
        self.U=np.zeros([p,batch_size])
        self.U[pp,[range(0,batch_size)]]=self.gdf(t,del_t)
    def x_rate_calc(self,x,p,batch_size):
        x_rate=np.zeros([p,9,batch_size])
        x_rate[:,0,:]=x[:,5,:]-x[:,6,:]#[mV]
        x_rate[:,1,:]=x[:,4,:]#[mV]
        x_rate[:,2,:]=x[:,5,:]#[mV]
        x_rate[:,3,:]=x[:,6,:]#[mV]
        x_rate[:,7,:]=x[:,8,:]#[mV]
        #print self.cU
        A4=(np.matmul((self.cF+self.cL+self.gamma_1*self.I),self.S(x[:,0,:]))+np.multiply(self.cU,self.U))#[mV/s]
        B4=np.divide(self.He,self.tau_e)#[]
        C4=2*np.divide(x[:,4,:],self.tau_e)#[mV/s]
        D4=np.divide(x[:,1,:],self.tau_e*self.tau_e)#[mV/s]
        x_rate[:,4,:]=np.multiply(B4,A4)-C4-D4#[mV/s]


        A5=(np.matmul((self.cB+self.cL),self.S(x[:,0,:]))+self.gamma_2*self.S(x[:,1,:]))#[mV/s]
        B5=np.divide(self.He,self.tau_e)#[]
        C5=2*np.divide(x[:,5,:],self.tau_e)#[mV/s]
        D5=np.divide(x[:,2,:],self.tau_e*self.tau_e)#[mV/s]
        x_rate[:,5]=np.multiply(B5,A5)-C5-D5#[mV/s]

        A6=(np.matmul(self.gamma_4*self.I,self.S(x[:,7,:])))#.reshape(p,batch_size)#[mV/s]
        B6=np.divide(self.Hi,self.tau_i)#[]
        C6=2*np.divide(x[:,6,:],self.tau_i)#[mV/s]
        D6=np.divide(x[:,3,:],self.tau_i*self.tau_i)#[mV/s]
        x_rate[:,6,:]=np.multiply(B6,A6)-C6-D6#[mV/s]

        A8=(np.matmul(self.cB+self.cL+self.gamma_3*self.I,self.S(x[:,0,:])))#[mV/s]
        B8=np.divide(self.He,self.tau_e)#[]
        C8=2*np.divide(x[:,8,:],self.tau_e)#[mV/s]
        D8=np.divide(x[:,7,:],self.tau_e*self.tau_e)#[mV/s]
        x_rate[:,8,:]=np.multiply(B8,A8)-C8-D8#[mV/s]
    
        return x_rate

    def x_loop(self,n_steps,del_t,p,batch_size,q_init):
        self.x0b=np.zeros([p,n_steps,batch_size])
        x=np.zeros([p,9,batch_size])
        for step in range(0,n_steps):
            t=del_t*step
            self.x0b[:,step,:]=x[:,0,:]
            self.u(t,p,q_init,batch_size,del_t)
            x=self.x_rate_calc(x,p,batch_size)*del_t+x
            
        self.x0b = np.transpose(self.x0b,(0,2,1))#p x batch_size x n_steps
        self.x0b = self.x0b.reshape([p,-1])#for use in fields_gen batch p x batch_size*n_steps
