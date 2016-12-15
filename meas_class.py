import numpy as np
from numpy import matlib
import sphere
import scipy
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def scale_dipole(dipole_in):
    #dipole_in is pxnxb
    p=dipole_in.shape[0]
    n=dipole_in.shape[1]
    b=dipole_in.shape[2]
    print p, " Dipoles"
    qq=dipole_in.transpose((0,2,1)).reshape([p,-1])#pxnxb->pxbxn->pxb*n
    #qq is pxbatch_size*n_steps - convert to one-hot for tf neural net
    #this should not be done if you want to fit current density.
    #ppp=np.argmax(np.abs(np.nan_to_num(qq)),axis=0)
    #q = np.zeros(qq.shape)
    #q[ppp,[range(0,qq.shape[1])]]=1.0#qq is pxbatch_size*n_steps
    #ONE-HOT!!! convert to one-hot for tf neural net
    #don't do that if fitting current density. in fact it's redundant if you use cross-entropy as the cost function.
    #(qq-np.amin(qq,axis=0))/(np.amax(qq,axis=0)-np.amin(qq,axis=0)))
    print "Dipoles now reshaped for TF NN."#as one-hot encoding"
    q=qq
    qtrue=q.reshape([p,b,n]).transpose((1,2,0))#bxnxp
    return qtrue, p

#preprocesses real data into a format acceptable to the NN.
#batched
#run __init__-->pca-->interp-->reshape
class meas:
    def __init__(self,meg_meas_in,meg_meas_xyz, eeg_meas_in,eeg_meas_xyz, meas_dims, n_steps, batch_size):
        #meas_in bxmxn
        self.m0 = meg_meas_in.shape[1]
        print self.m0, " MEG sensors"
        self.m1 = eeg_meas_in.shape[1]
        print self.m1, " EEG sensors"
        MEG=np.transpose(meg_meas_in,(1,0,2)).reshape([self.m0,-1])
        EEG=np.transpose(eeg_meas_in,(1,0,2)).reshape([self.m1,-1])
        self.meas_in = [MEG, EEG]#[m0|m1]xbatch_size*n_steps
        self.meas_xyz = [meg_meas_xyz, eeg_meas_xyz]#m0x3,m1x3
        self.meas_dims= meas_dims
        self.n_steps=n_steps
        self.batch_size=batch_size
    def grid_dims(self):
        print "TF image grid ..."
        xmin = np.amin([np.amin(self.X0[0]),np.amin(self.X0[1])])
        ymin = np.amin([np.amin(self.Y0[0]),np.amin(self.Y0[1])])
        xmax = np.amin([np.amax(self.X0[0]),np.amax(self.X0[1])])
        ymax = np.amin([np.amax(self.Y0[0]),np.amax(self.Y0[1])])
        print xmin,xmax,ymin,ymax

        azmin = np.amin([np.amin(self.AZ0[0]),np.amin(self.AZ0[1])])
        elmin = np.amin([np.amin(self.EL0[0]),np.amin(self.EL0[1])])
        azmax = np.amin([np.amax(self.AZ0[0]),np.amax(self.AZ0[1])])
        elmax = np.amin([np.amax(self.EL0[0]),np.amax(self.EL0[1])])
        print azmin,azmax,elmin,elmax

        
        self.x = np.linspace(xmin,xmax,self.meas_dims[0])
        self.y = np.linspace(ymin,ymax,self.meas_dims[1])

        
        self.az = np.linspace(azmin,azmax,self.meas_dims[0])
        self.el = np.linspace(elmin,elmax,self.meas_dims[1])

        self.X,self.Y = np.meshgrid(self.x,self.y)

        self.AZ,self.EL = np.meshgrid(self.az,self.el)

        self.m=np.size(self.X)

    def grid_xyz(self):
        #Observation grid
        print "Real observation grid ..."
        x0,y0,z0=np.split(self.meas_xyz[0],3,1)
        x1,y1,z1=np.split(self.meas_xyz[1],3,1)

        az0, el0, r0=sphere.cart2sph(x0,y0,z0)
        az1, el1, r1=sphere.cart2sph(x1,y1,z1)
        
        self.X0=[x0,x1]
        self.Y0=[y0,y1]
        self.Z0=[z0,z1]
        
        self.AZ0=[az0,az1]
        self.EL0=[el0,el1]
        self.R0=[r0,r1]

    def pca(self):
        for channel in [0,1]:
            mPCA=PCA(self.meas_in[channel])
            self.meas_in[channel]=mPCA.Y

    def scale(self):
        for channel in [0,1]:
            self.meas_in[channel]=np.nan_to_num((self.meas_in[channel]-np.amin(self.meas_in[channel],axis=0))/(np.amax(self.meas_in[channel],axis=0)-np.amin(self.meas_in[channel],axis=0)))
            
    def interp(self):
        self.grid_xyz()
        self.grid_dims()#make grid to interpolate

        print "Interpolate ..."
        f=np.zeros((self.m,self.n_steps*self.batch_size))
        g=np.zeros((self.m,self.n_steps*self.batch_size))
        self.meas_out=[f,g]
        for channel in [0,1]:
            print "Channel ", channel
            for step in range(0,self.n_steps*self.batch_size):
                #print "Generate interp function for step ", step
                a=np.squeeze(self.AZ0[channel])
                b=np.squeeze(self.EL0[channel])
                c=np.squeeze(self.meas_in[channel][:,step])
                out=scipy.interpolate.griddata((a,b),c,(self.AZ,self.EL),method='nearest')
                #print "Apply interp function for step ", step
                self.meas_out[channel][:,step]=out.ravel()
                #mxbatch_size*n_steps

    def reshape(self):
        self.meas_img=np.array(self.meas_out).reshape([2,-1,self.batch_size,self.n_steps])#2xmxbatchsizexnsteps
        self.meas_img=np.transpose(self.meas_img,(2,3,0,1))#batch_sizexnstepsx2xm
        self.meas_img=self.meas_img.reshape([self.batch_size,self.n_steps,2,self.meas_dims[1],self.meas_dims[0]])#bxnx2xMxN (m=MXN)
        self.meas_img=np.squeeze(np.transpose(self.meas_img,(0,1,4,3,2)))#proper format for tf nn. bxnxNxMx2

    def plot(self, step):

        fig = plt.figure()
        b,n=np.unravel_index(step,[self.batch_size,self.n_steps])

        for channel in [0,1]:
            print "plotting interp"
            fplot=np.squeeze(self.meas_out[channel][:,step])#m
            ax = fig.add_subplot(2,2,1+2*channel, projection='3d')

            ax.plot_trisurf(np.squeeze(self.AZ.ravel()), np.squeeze(self.EL.ravel()), fplot, cmap=cm.jet, linewidth=0.2)
            
            print "plotting real" 
            fplot=np.squeeze(self.meas_in[channel][:,step])#m

            ax = fig.add_subplot(2,2,2+2*channel, projection='3d')

            ax.plot_trisurf(np.squeeze(self.AZ0[channel]), np.squeeze(self.EL0[channel]), fplot, cmap=cm.jet, linewidth=0.2)

        plt.show()
            
        
