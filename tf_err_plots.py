import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

data=np.zeros([1,12])
csvfile = open('./nn_real.csv','r')
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
            col_n_layer=header.index("n_layer")        
            col_n_step=header.index("step")         
            col_rmse=header.index("rmse")         
            col_acc=header.index("accuracy")         
            col_ce=header.index("xentropy")         
            col_rmse_last=header.index("rmse_last")         
            col_acc_last=header.index("accuracy_last")         
            col_ce_last=header.index("xentropy_last")     
        else:
             
            this=np.array([row[col_cost], row[col_cost_step], row[n_conv1], row[n_conv2],row[n_layer], row[n_lstm], row[step], row[col_rmse],row[col_acc], row[col_ce], row[col_rmse_last],row[col_acc_last], row[col_ce_last]])
            this=this.reshape([12,1]).T
            data=np.append(data,this,axis=0)
        
        rownum+=1
finally:
    csvfile.close()

col_cost=0
col_cost_step=1
col_n_conv1=2
col_n_conv2=3
col_n_layer=4        
col_n_step=5         
col_rmse=6         
col_acc=7         
col_ce=8         
col_rmse_last=9        
col_acc_last=10       
col_ce_last=11   

cost_vals=['cross','rmse']
cost_step_vals=['cross','rmse']
n_conv1_vals=[2, 5, 10]
n_conv1_vals=[2, 5, 10]

n_layer_vals=[2, 3]
n_lstm_vals=[100,300,1000]

data=np.delete(data,(0),axis=0)
colors=('b', 'g', 'r', 'k')
for e in [col_acc, col_acc_last]:
    for a1 in cost_vals:
        for a2 in cost_step_vals:
            idx = 0
            for n1 in n_conv1_vals:
                for n2 in n_conv2_vals:
                    idx+=1 

                    plt.subplot(3, 3, idx)
                    idx2=0
                    for m1 in n_layers_vals:
                        for m2 in n_lstm_vals:
                            idx2+=1
                            col=colors(idx2)
                            lab='n_layers='+str(m1)+', n_lstm='+str(m2)
                            picks = np.where(data[:,col_cost]==a1)
                            picks = np.intersect1d(picks, np.where(data[:,col_cost_step]==a2))
                            picks = np.intersect1d(picks, np.where(data[:,col_n_conv1]==n1))
                            picks = np.intersect1d(picks, np.where(data[:,col_n_conv2]==n2))
                            picks = np.intersect1d(picks, np.where(data[:,col_n_layers]==m1))
                            picks = np.intersect1d(picks, np.where(data[:,col_n_lstm]==m2))
                            data_slice_x=data[picks,col_step]
                            data_slice_y=data[picks,e]
                            plt.plot(data_slice_x.T, data_slice_y.T,col,label=lab)
                    plt.title('a1'+', '+a2)
                    plt.xlim(0,np.amax(data_slice_x))
                    if e==col_acc:
                        plt.ylabel('Accuracy')
                    elif e==col_acc_last:
                        plt.ylabel('Accuracy (last step only)')

                    if idx==9:
                        legend = plt.legend(loc='lower right')
                        plt.xlabel('Step')
                        plt.show()

