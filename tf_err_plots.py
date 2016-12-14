import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fieldnames=['fixed','noise','batches','learning rate','batch_size','per_batch','dropout','k_conv','n_conv1','n_conv2','n_layer','n_steps','train step','xentropy','rmse','accuracy']

data=np.zeros([1,8])
csvfile = open('./nn_whitening_errors_noise_free.csv','r')
try:
    reader=csv.reader(csvfile)
    rownum=0
    for row in reader:
        if rownum==0:
            header=row
            col_fixed=header.index("fixed")
            col_noise=header.index("noise")
            col_batches=header.index("batches")
            col_rate=header.index("learning rate")
            col_size=header.index("batch_size")        
            col_per=header.index("per_batch")
            col_step=header.index("train step")
            col_nstep=header.index("n_steps")
            col_x=header.index("xentropy")
            col_rmse=header.index("rmse")
            col_acc=header.index("accuracy")
        else:
            #print row[col_fixed], row[col_batches], row[col_rate], row[col_size], row[col_per], row[col_step], row[col_size], row[col_per], row[col_step], row[col_x], row[col_rmse],row[col_acc]
            this=np.array([row[col_fixed], row[col_noise], row[col_per], row[col_nstep],row[col_step], row[col_x], row[col_rmse],row[col_acc]])
            this=this.reshape([8,1]).T
            data=np.append(data,this,axis=0)
        
        rownum+=1
finally:
    csvfile.close()

data=np.delete(data,(0),axis=0)

col_fixed=0
col_noise=1
col_per=2
col_nstep=3
col_step=4
col_x=5
col_rmse=6
col_acc=7
fixed=np.where(data[:,col_fixed]=='True')
data[fixed,col_fixed]=1
free=np.where(data[:,col_fixed]=='False')
data[free,col_fixed]=0

noise=np.where(data[:,col_noise]=='True')
data[noise,col_noise]=1
nonoise=np.where(data[:,col_noise]=='False')
data[nonoise,col_noise]=0

data=np.float32(data)

p100=np.where(data[:,col_per]==100.)
p500=np.where(data[:,col_per]==500.)

n1=np.where(data[:,col_nstep]==1.)
n100=np.where(data[:,col_nstep]==100.)

train=np.where(data[:,col_step]>-1.)
test=np.where(data[:,col_step]==-1.)

fixed_noise_100_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,noise),p100),n1),train),4:8]
free_noise_100_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,noise),p100),n1),train),4:8]

fixed_nonoise_100_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,nonoise),p100),n1),train),4:8]
free_nonoise_100_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,nonoise),p100),n1),train),4:8]

fixed_noise_500_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,noise),p500),n1),train),4:8]
free_noise_500_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,noise),p500),n1),train),4:8]

fixed_nonoise_500_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,nonoise),p500),n1),train),4:8]
free_nonoise_500_1=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,nonoise),p500),n1),train),4:8]

fixed_noise_100_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,noise),p100),n100),train),4:8]
free_noise_100_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,noise),p100),n100),train),4:8]

fixed_nonoise_100_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,nonoise),p100),n100),train),4:8]
free_nonoise_100_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,nonoise),p100),n100),train),4:8]

fixed_noise_500_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,noise),p500),n100),train),4:8]
free_noise_500_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,noise),p500),n100),train),4:8]

fixed_nonoise_500_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(fixed,nonoise),p500),n100),train),4:8]
free_nonoise_500_100=data[np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(free,nonoise),p500),n100),train),4:8]

plt.subplot(4, 2, 1)
plt.plot(fixed_noise_100_1[:,0],fixed_noise_100_1[:,3],'-b',free_noise_100_1[:,0],free_noise_100_1[:,3],'-r')
plt.title('Noise, 100 train steps, 1 time steps')
plt.xlim(0,np.amax(fixed_noise_100_1[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 2)
plt.plot(fixed_noise_100_100[:,0],fixed_noise_100_100[:,3],'-b',free_noise_100_100[:,0],free_noise_100_100[:,3],'-r')
plt.title('Noise, 100 train steps, 100 time steps')
plt.xlim(0,np.amax(fixed_noise_100_100[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 3)
plt.plot(fixed_noise_500_100[:,0],fixed_noise_500_100[:,3],'-b',free_noise_500_100[:,0],free_noise_500_100[:,3],'-r')
plt.title('Noise, 500 train steps, 100 time steps')
plt.xlim(0,np.amax(fixed_noise_500_100[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 4)
plt.plot(fixed_noise_500_1[:,0],fixed_noise_500_1[:,3],'-b',free_noise_500_1[:,0],free_noise_500_1[:,3],'-r')
plt.title('Noise, 500 train steps, 1 time steps')
plt.xlim(0,np.amax(fixed_noise_500_1[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 5)
plt.plot(fixed_nonoise_100_1[:,0],fixed_nonoise_100_1[:,3],'-b',free_nonoise_100_1[:,0],free_nonoise_100_1[:,3],'-r')
plt.title('Nonoise, 100 train steps, 1 time steps')
plt.xlim(0,np.amax(fixed_nonoise_100_1[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 6)
plt.plot(fixed_nonoise_100_100[:,0],fixed_nonoise_100_100[:,3],'-b',free_nonoise_100_100[:,0],free_nonoise_100_100[:,3],'-r')
plt.title('Nonoise, 100 train steps, 100 time steps')
plt.xlim(0,np.amax(fixed_nonoise_100_100[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 7)
plt.plot(fixed_nonoise_500_100[:,0],fixed_nonoise_500_100[:,3],'-b',free_nonoise_500_100[:,0],free_nonoise_500_100[:,3],'-r')
plt.title('Nonoise, 500 train steps, 100 time steps')
plt.xlim(0,np.amax(fixed_nonoise_500_100[:,0]))
plt.ylabel('Accuracy')

plt.subplot(4, 2, 8)
plt.plot(fixed_nonoise_500_1[:,0],fixed_nonoise_500_1[:,3],'-b',label='fixed')
plt.plot(free_nonoise_500_1[:,0],free_nonoise_500_1[:,3],'-r',label='free')
plt.title('Nonoise, 500 train steps, 1 time steps')
plt.xlim(0,np.amax(fixed_nonoise_500_1[:,0]))
plt.ylabel('Accuracy')

legend = plt.legend(loc='lower right')
plt.xlabel('Step')
plt.show()


test_results=np.squeeze(data[test,:])
print test_results
