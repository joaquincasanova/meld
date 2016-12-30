import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

data=np.zeros([1,9])
csvfile = open('./nn_real_rnn_ttv.csv','r')
try:
    reader=csv.reader(csvfile)
    rownum=0
    for row in reader:
        print row
        print len(row)
        #time.sleep(1)
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
            try: 
                this=np.array([np.float(row[col_step]),np.float(row[col_n_layer]), np.float(row[col_n_lstm]),  np.float(row[col_rmse]),np.float(row[col_acc]), np.float(row[col_ce]), np.float(row[col_rmse_last]),np.float(row[col_acc_last]), np.float(row[col_ce_last])])
                this=this.reshape([9,1]).T
                data=np.append(data,this,axis=0)
            except:
                break
        rownum+=1
            
finally:
    csvfile.close()

col_n_lstm=2
col_n_layer=1
col_step=0
col_rmse=3         
col_acc=4        
col_ce=5        
col_rmse_last=6        
col_acc_last=7       
col_ce_last=8   

lstm_vals=[100,300,1000]
layer_vals=[1,2]

err_col_vals=[3,4,5,6,7,8]

err_col_lab=['RMSE','Accuracy','Cross Entropy','RMSE Last','Accuracy Last','Cross Entropy Last']

data=np.delete(data,(0),axis=0)
colors=('b', 'g', 'r', 'k', 'y', 'c')
idx = 0
for ls in lstm_vals:
    for la in layer_vals:
        idx+=1
        edx=0
        for e in err_col_vals:
            edx+=1 

            plt.subplot(3, 3, edx)
        
            col=colors[idx-1]
            lab='n_lstm='+str(ls)+', n_layers='+str(la)
            picks = np.where(abs(data[:,col_n_lstm]-ls)<.01)
            picks = np.intersect1d(picks, np.where(abs(data[:,col_n_layer]-la)<.01))
            picks = np.intersect1d(picks, np.where(data[:,col_step]>-1.0))
            #print len(picks)
            #time.sleep(1)
            data_slice_x=data[picks,col_step]
            data_slice_y=data[picks,e]
            plt.plot(data_slice_x.T, data_slice_y.T,col,label=lab)
            
            plt.xlim(0,500*5.)
            plt.ylabel(err_col_lab[edx-1])
            if edx==4:
                plt.xlabel('Step')
                
            if edx==5:
                legend = plt.legend(loc=(0,-1.),labelspacing=0)
plt.show()

