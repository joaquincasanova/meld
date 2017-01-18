import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

data=np.zeros([1,9])
csvfile = open('./nn_real_locate_faces_7.csv','r')
try:
    reader=csv.reader(csvfile)
    rownum=0
    for row in reader:
        #print row
        #print len(row)
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
            col_beta=header.index("beta")
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

lstm_vals=[10,30,100]
layer_vals=[1,2]

err_col_vals=[3,5]#[3,4,5,6,7,8]

err_col_lab=['RMSE','RMSE Last']#['RMSE','Accuracy','Cross Entropy','RMSE Last','Accuracy Last','Cross Entropy Last']

data=np.delete(data,(0),axis=0)
colors=('b', 'g', 'r', 'k', 'y', 'c')
idx = 0
for ls in lstm_vals:
    for la in layer_vals:
        idx+=1
        edx=0
        for e in err_col_vals:
            edx+=1 

            plt.subplot(3, 1, edx)
        
            col=colors[idx-1]
            lab='n_lstm='+str(ls)+', n_layers='+str(la)
            print lab
            picks = np.where(data[:,col_n_lstm]==ls)
            picks = np.intersect1d(picks, np.where(data[:,col_n_layer]==la))
            picks = np.intersect1d(picks, np.where(data[:,col_step]>=0))
            test = np.where(data[:,col_n_lstm]==ls)
            test = np.intersect1d(test, np.where(data[:,col_n_layer]==la))
            test = np.intersect1d(test, np.where(data[:,col_step]<=-2))
            test_acc_last=np.mean(data[test,5])
            val= np.where(data[:,col_n_lstm]==ls)
            val = np.intersect1d(val, np.where(data[:,col_n_layer]==la))
            val = np.intersect1d(val, np.where(data[:,col_step]==-1))
            print data[val,5]
            val_acc_last=np.mean(data[val,5])
            lab='n_lstm='+str(ls)+', n_layers='+str(la)+', val rmse='+str(val_acc_last)+', test rmse='+str(test_acc_last)
             
            #print len(picks)
            #time.sleep(1)
            data_slice_x=data[picks,col_step].reshape([1,-1])
            
            data_slice_xp=np.arange(0,data_slice_x.shape[1]).reshape([1,-1])
            data_slice_y=data[picks,e].reshape([1,-1])
            plt.plot(data_slice_xp.T, data_slice_y.T,col,label=lab)
            
            plt.xlim(0,500*20.)
            plt.ylabel(err_col_lab[edx-1])
            if edx==2:
                plt.xlabel('Step')
                
            if edx==2:
                legend = plt.legend(loc=(0,-1.),labelspacing=0)
plt.show()

