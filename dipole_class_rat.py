
import numpy as np
from numpy import matlib
import sphere
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

class dipole:
    def __init__(self, delT, batch_size, n_steps, n_chan_in, meas_dims, dipole_dims, orient=None, noise_flag=None,dipole_xyz=None, meas_xyz=None, pca=True):
        self.delT=delT
        self.n_chan_in=n_chan_in
        self.noise_flag=noise_flag
        if n_steps is None:
            self.n_steps=1
        else:
            self.n_steps=n_steps

        if meas_xyz is None: 
            self.meas_dims=meas_dims 
            self.meas_xyz=meas_xyz
        else:
            self.meas_dims=meas_dims
            #measurement coordinates given. use meas_dims to interpolate results to image format for CNN input
            self.meas_xyz=meas_xyz
            
        if dipole_xyz is not None:
            self.dipole_xyz=dipole_xyz
            self.dipole_dims=dipole_dims
            self.grid_xyz()
        else:
            self.dipole_dims=dipole_dims
            self.grid_dims()

        #print dipole_dims, "Dipole dims"

        #print self.p, "Dipoles (dipole_class_rat)"
            
        self.batch_size=batch_size
        self.mu0=4*np.pi*1e-7
        self.orient=orient#use True if you want each example to have different orientations
        self.pca=pca
        self.dipole_init()
        self.GB,self.GB2 = self.p_dipole_gain_B()#Mosher-style gain matrix
        self.GV,self.GV2 = self.p_dipole_gain_V()#Mosher-style ga
        self.dipole_orient()

        #print dipole_dims, "Dipole dims"

        #print self.p, "Dipoles (dipole_class_rat)"
        
    def dipole_init(self):    

        lnqq = np.random.standard_normal([self.p,self.batch_size,self.n_steps])
        self.qq = np.exp(lnqq)*100e-9#A/m
        #qq is pxb*n
        self.qq = self.qq.reshape([self.p,-1])
        
    def dipole_orient(self):
        if self.orient is False:
            self.qx=np.random.rand(self.p,1)
            self.qy=np.random.rand(self.p,1)
            self.qz=np.random.rand(self.p,1)
            self.qmag=np.sqrt(self.qx**2+self.qy**2+self.qz**2)
            self.qx=self.qx/self.qmag
            self.qy=self.qy/self.qmag
            self.qz=self.qz/self.qmag
            self.dipole_orient=np.hstack([self.qx,self.qy,self.qz])
        else:
            self.qx=np.random.rand(self.p,self.batch_size)
            self.qy=np.random.rand(self.p,self.batch_size)
            self.qz=np.random.rand(self.p,self.batch_size)
            self.qmag=np.sqrt(self.qx**2+self.qy**2+self.qz**2)
            self.qx=self.qx/self.qmag
            self.qy=self.qy/self.qmag
            self.qz=self.qz/self.qmag
            self.dipole_orient=np.hstack([self.qx,self.qy,self.qz])
            
    def grid_dims(self):
        self.r = 15e-3#15mm    
        self.el = np.linspace(0,np.pi,self.meas_dims[0])
        self.az = np.linspace(2*np.pi/self.meas_dims[1],2*np.pi,self.meas_dims[1])

        self.EL,self.AZ=np.meshgrid(self.el,self.az)
        #for plotting
        self.EL0=self.EL
        self.AZ0=self.AZ

        #Observation grid
        self.AZ = self.AZ.reshape(np.size(self.AZ),1)
        self.EL = self.EL.reshape(np.size(self.EL),1)
        self.R = self.r*np.ones([np.size(self.EL),1])

        self.X, self.Y, self.Z=sphere.sph2cart(self.AZ,self.EL,self.R)#Sensor locations (cartesian)
        self.meas_xyz = np.hstack((np.hstack((self.X,self.Y)),self.Z))
        self.m=np.size(self.X)

        #Dipole grid
        
        self.rq = np.linspace(14e-3/self.dipole_dims[0],14e-3,self.dipole_dims[0])#14mm
        self.elq = np.linspace(np.pi/2-np.pi/2/self.dipole_dims[1],-np.pi/2+np.pi/2/self.dipole_dims[1],self.dipole_dims[1])
        self.azq = np.linspace(2*np.pi/self.dipole_dims[2],2*np.pi,self.dipole_dims[2])
        self.RQ,self.ELQ,self.AZQ=np.meshgrid(self.rq,self.elq,self.azq)

        self.p=np.size(self.RQ)

        self.AZQ = self.AZQ.reshape(self.p,1)
        self.ELQ = self.ELQ.reshape(self.p,1)
        self.RQ = self.RQ.reshape(self.p,1)
        
        self.xq,self.yq,self.zq = sphere.sph2cart(self.AZQ,self.ELQ,self.RQ)#dipole locations (cartesian)
        self.dipole_xyz = np.hstack((np.hstack((self.xq,self.yq)),self.zq))
        
    def grid_xyz(self):      
        self.m=self.meas_xyz.shape[0]       
        self.p=self.dipole_xyz.shape[0]
        #for plotting

        #Observation grid        
        self.X,self.Y,self.Z=np.split(self.meas_xyz,3,1)
        self.AZ,self.EL,self.R=sphere.cart2sph(self.X,self.Y,self.Z)

        #Dipole grid
        self.xq,self.yq,self.zq=np.split(self.dpole_xyz,3,1) 
        self.AZQ,self.ELQ,self.RQ=sphere.cart2sph(self.xq,self.yq,self.zq) 
        
    def dipole_gain_B(self,X0,Y0,Z0,AZ0,EL0,R0):
        #returns gain matrix for a dipole at R0. following mosher 1992.
        xyzRR0=np.ones([self.m,3])
        xyzRR0[:,0]=self.X[:,0]-X0[:,0]
        xyzRR0[:,1]=self.Y[:,0]-Y0[:,0]
        xyzRR0[:,2]=self.Z[:,0]-Z0[:,0]
        uno=np.ones([self.m,1])
        x,y,z = sphere.sph2cart(self.AZ,self.EL,uno)
        xyzr=np.ones([self.m,3])
        xyzr[:,0]=x[:,0]
        xyzr[:,1]=y[:,0]
        xyzr[:,2]=z[:,0]
        numerator = np.cross(xyzRR0,xyzr)#(mx3)
        denominator = np.transpose(np.matlib.repmat(sphere.vect_magnitude(xyzRR0)**3,3,1))
        G = numerator/denominator*self.mu0/4/np.pi
        Gx=np.transpose([G[:,0],])
        Gy=np.transpose([G[:,1],])
        Gz=np.transpose([G[:,2],])

        Gaz, Gel ,Gr = sphere.vect_cart2sph(Gx,Gy,Gz,AZ0,EL0)
        G2 = np.hstack([Gaz,Gel])
        return G,G2

    def p_dipole_gain_B(self):

        #R is mx3 list of sensor locations
        #R0p is px3 list of dipole locations
        #Returns mx3p gain matrix and mx2p gain matrix
        G=np.ones([self.m,3*self.p])
        G2=np.ones([self.m,2*self.p])
        for p_ind in range(0,self.p):        
            X0=np.matlib.repmat(self.xq[p_ind,0],self.m,1)
            Y0=np.matlib.repmat(self.yq[p_ind,0],self.m,1)
            Z0=np.matlib.repmat(self.zq[p_ind,0],self.m,1)
            AZ0=np.matlib.repmat(self.AZQ[p_ind,0],self.m,1)
            EL0=np.matlib.repmat(self.ELQ[p_ind,0],self.m,1)
            R0=np.matlib.repmat(self.RQ[p_ind,0],self.m,1)
            g,g2=self.dipole_gain_B(X0,Y0,Z0,AZ0,EL0,R0)
            G[:,3*p_ind:3*p_ind+3] = g
            G2[:,2*p_ind:2*p_ind+2] = g2
        return G,G2

    def dipole_gain_V(self,X0,Y0,Z0,AZ0,EL0,R0):
        #returns gain matrix for a dipole at R0. following mosher 1992.
        sigma=0.3 #mho/m
        
        xyzRR0=np.ones([self.m,3])
        xyzRR0[:,0]=self.X[:,0]-X0[:,0]
        xyzRR0[:,1]=self.Y[:,0]-Y0[:,0]
        xyzRR0[:,2]=self.Z[:,0]-Z0[:,0]
        numerator = xyzRR0#(mx3)
        denominator = np.transpose(np.matlib.repmat(sphere.vect_magnitude(xyzRR0)**3,3,1))
        G = numerator/denominator/sigma/4/np.pi
        Gx=np.transpose([G[:,0],])
        Gy=np.transpose([G[:,1],])
        Gz=np.transpose([G[:,2],])

        Gaz, Gel ,Gr = sphere.vect_cart2sph(Gx,Gy,Gz,AZ0,EL0)
        G2 = np.hstack([Gaz,Gel])
        return G,G2

    def p_dipole_gain_V(self):

        #R is mx3 list of sensor locations
        #R0p is px3 list of dipole locations
        #Returns mx3p gain matrix and mx2p gain matrix
        G=np.ones([self.m,3*self.p])
        G2=np.ones([self.m,2*self.p])
        for p_ind in range(0,self.p):        
            X0=np.matlib.repmat(self.xq[p_ind,0],self.m,1)
            Y0=np.matlib.repmat(self.yq[p_ind,0],self.m,1)
            Z0=np.matlib.repmat(self.zq[p_ind,0],self.m,1)
            AZ0=np.matlib.repmat(self.AZQ[p_ind,0],self.m,1)
            EL0=np.matlib.repmat(self.ELQ[p_ind,0],self.m,1)
            R0=np.matlib.repmat(self.RQ[p_ind,0],self.m,1)
            g,g2=self.dipole_gain_V(X0,Y0,Z0,AZ0,EL0,R0)
            G[:,3*p_ind:3*p_ind+3] = g
            G2[:,2*p_ind:2*p_ind+2] = g2
        return G,G2

    def fields_gen_batch(self):
            
        if self.orient is False:
            Qx=self.qx*self.qq#px1 times pxbatch_size*n_steps, elementwise
            Qy=self.qy*self.qq
            Qz=self.qz*self.qq

            Q=np.vstack([Qx,Qy,Qz])#3xpxbatch_size*n_steps
            Q=np.float32(Q.reshape(3*self.p,-1))#3pxbatch_size*n_steps
        else:
            Qx=np.zeros([self.p,self.batch_size,self.n_steps])
            Qy=np.zeros([self.p,self.batch_size,self.n_steps])
            Qz=np.zeros([self.p,self.batch_size,self.n_steps])
            self.qq=self.qq.reshape([self.p,self.batch_size,self.n_steps])
            for step in range(0,self.n_steps):
                Qx[:,:,step]=self.qx*self.qq[:,:,step]#pxbatch_size times pxbatch_sizexn_steps, elementwise
                Qy[:,:,step]=self.qy*self.qq[:,:,step]
                Qz[:,:,step]=self.qz*self.qq[:,:,step]

            Q=np.vstack([Qx,Qy,Qz])#3xpxbatch_sizexn_steps
            Q=np.float32(Q.reshape(3*self.p,-1))#3pxbatch_size*n_steps
            
            self.qq=self.qq.reshape([self.p,self.batch_size*self.n_steps])#pxbatch_size*n_steps
            
        bGen=np.matmul(self.GB,Q)
        vGen=np.matmul(self.GV,Q)
        meas=np.float32(np.vstack([bGen,vGen]))
        self.bmeas=np.float32(bGen)
        self.vmeas=np.float32(vGen)
        
    def batch_sequence_gen(self):
        if self.noise_flag is None or self.noise_flag is False:
            bandwidth=0.#/self.delT
        else:
            bandwidth=100.0
            
        noise_sigma_B = np.sqrt(((1e-12)**2)*bandwidth)  
        noise_sigma_V = np.sqrt(((1e-9)**2)*bandwidth)
        #print noise_sigma_B, noise_sigma_V
        self.fields_gen_batch()      

        self.qtrue=self.qq        
        self.bmeas = self.bmeas + noise_sigma_B*np.random.randn(self.bmeas.shape[0],self.bmeas.shape[1])
        self.vmeas = self.vmeas + noise_sigma_V*np.random.randn(self.vmeas.shape[0],self.vmeas.shape[1])

        self.eeg_data = np.transpose(self.vmeas.reshape([-1,self.batch_size,self.n_steps]),(1,0,2))#bxmxn
        self.meg_data = np.transpose(self.bmeas.reshape([-1,self.batch_size,self.n_steps]),(1,0,2))#bxmxn
            
        self.qtrue=np.transpose(self.qtrue.reshape([-1,self.batch_size,self.n_steps]),(0, 2, 1))#pxnxb

        self.eeg_xyz = self.meas_xyz
        self.meg_xyz = self.meas_xyz
