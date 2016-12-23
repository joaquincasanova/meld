import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

data=np.zeros([1,9])
csvfile = open('./nn_real_rnn_wd.csv','r')
try:
    reader=csv.reader(csvfile)
    rownum=0
    for row in reader:
        if rownum==0:
            header=row
            col_cost=header.index("cost")
            col_cost_step=header.index("cost_step")
            col_n_conv1=header.index("n_conv1")
            col_n_conv2=header.index("n_conv2")
            col_batches=header.index("batches")
            col_rate=header.index("learning rate")
            col_per=header.index("per_batch")
            col_drop=header.index("dropout")
            col_cost=header.index("cost")
            col_n_lstm=header.index("n_lstm")
            col_n_layer=header.index("n_layer")        
            col_step=header.index("train step")        
            col_n_steps=header.index("n_steps")         
            col_rmse=header.index("rmse")         
            col_acc=header.index("accuracy")         
            col_ce=header.index("xentropy")         
            col_rmse_last=header.index("rmse_last")         
            col_acc_last=header.index("accuracy_last")         
            col_ce_last=header.index("xentropy_last")     
        else:
             
            this=np.array([row[col_cost], row[col_cost_step], np.float(row[col_step]), np.float(row[col_rmse]),np.float(row[col_acc]), np.float(row[col_ce]), np.float(row[col_rmse_last]),np.float(row[col_acc_last]), np.float(row[col_ce_last])])
            this=this.reshape([9,1]).T
            data=np.append(data,this,axis=0)
        
        rownum+=1
finally:
    csvfile.close()

col_cost=0
col_cost_step=1
col_step=2
col_rmse=3         
col_acc=4        
col_ce=5        
col_rmse_last=6        
col_acc_last=7       
col_ce_last=8   

cost_vals=['cross','rmse']
cost_step_vals=['all','last']

err_col_vals=[3,4,5,6,7,8]

err_col_lab=['RMSE','Accuracy','Cross Entropy','RMSE Last','Accuracy Last','Cross Entropy Last']

data=np.delete(data,(0),axis=0)
colors=('b', 'g', 'r', 'k')
idx = 0
for cf in cost_vals:
    for cs in cost_step_vals:
        idx+=1
        edx=0
        for e in err_col_vals:
            edx+=1 

            plt.subplot(3, 3, edx)
        
            col=colors[idx-1]
            lab='cost_function='+cf+', cost_step='+cs
            picks = np.where(data[:,col_cost]==cf)
            picks = np.intersect1d(picks, np.where(data[:,col_cost_step]==cs))
            picks = np.intersect1d(picks, np.where(data[:,col_step]!='-1.0'))
            data_slice_xs=data[picks,col_step]
            data_slice_ys=data[picks,e]
            data_slice_x=np.zeros(data_slice_xs.shape)
            data_slice_y=np.zeros(data_slice_ys.shape)
            for i in range(0,len(data_slice_x)):
                data_slice_x[i]=np.float(data_slice_xs[i])
                data_slice_y[i]=np.float(data_slice_ys[i])
            plt.plot(data_slice_x.T, data_slice_y.T,col,label=lab)
            
            plt.xlim(0,np.amax(data_slice_x))
            plt.ylabel(err_col_lab[edx-1])
            if edx==4:
                plt.xlabel('Step')
                
            if edx==5:
                legend = plt.legend(loc=(0,-1.),labelspacing=0)
plt.show()

