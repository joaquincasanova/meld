import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']
for train_id in [8,7]:
    for test_id in [7,8]:
        fname = './nn_real_locate_faces_cross_%s_%s.csv' % (train_id, test_id)

        data=np.zeros([1,11])
        csvfile = open(fname,'r')
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
                    col_size=header.index("batch_size")
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
                        this=np.array([np.float(row[col_step]),np.float(row[col_per]),np.float(row[col_size]),np.float(row[col_n_layer]), np.float(row[col_n_lstm]),  np.float(row[col_rmse]),np.float(row[col_acc]), np.float(row[col_ce]), np.float(row[col_rmse_last]),np.float(row[col_acc_last]), np.float(row[col_ce_last])])
                        this=this.reshape([11,1]).T
                        data=np.append(data,this,axis=0)
                    except:
                        continue
                rownum+=1

        finally:
            csvfile.close()

        col_step=0
        col_per=1
        col_size=2
        col_n_layer=3
        col_n_lstm=4
        col_rmse=5         
        col_acc=6        
        col_ce=7        
        col_rmse_last=8        
        col_acc_last=9       
        col_ce_last=10   

        lstm_vals=[10,25]
        layer_vals=[2,3]

        per_vals=[500,1000]
        size_vals=[50,100]

        err_col_vals=[5,8]#[3,4,5,6,7,8]

        err_col_lab=['RMSE','RMSE Last']#['RMSE','Accuracy','Cross Entropy','RMSE Last','Accuracy Last','Cross Entropy Last']

        data=np.delete(data,(0),axis=0)
        leg=[('-','b'), ('-','g'),('-', 'r'), ('-','k'),(':','b'), (':','g'),(':', 'r'), (':','k'),('--','b'), ('--','g'),('--', 'r'), ('--','k'),('-.','b'), ('-.','g'),('-.', 'r'), ('-.','k')]
        idx = 0
        for ls in lstm_vals:
            for la in layer_vals:
                for ns in size_vals:
                    for npr in per_vals:
                        idx+=1
                        edx=0
                        for e in err_col_vals:
                            edx+=1 

                            plt.subplot(3, 1, edx)

                            col=leg[idx-1][1]
                            lin=leg[idx-1][0]

                            lab='n_lstm='+str(ls)+', n_layers='+str(la)+', per='+str(npr)+', size='+str(ns)
                            print lab
                            picks = np.where(data[:,col_n_lstm]==ls)
                            print picks
                            picks = np.intersect1d(picks, np.where(data[:,col_n_layer]==la))
                            print picks
                            picks = np.intersect1d(picks, np.where(data[:,col_size]==ns))
                            print picks
                            picks = np.intersect1d(picks, np.where(data[:,col_per]==npr))
                            print picks
                            test = picks
                            val = picks
                            picks = np.intersect1d(picks, np.where(data[:,col_step]>=0))
                            test = np.intersect1d(test, np.where(data[:,col_step]<=-2))
                            test_acc_last=np.mean(data[test,8])
                            val = np.intersect1d(val, np.where(data[:,col_step]==-1))
                            print data[val,8]
                            val_acc_last=np.mean(data[val,8])
                            lab=lab+', val rmse='+str(val_acc_last)+', test rmse='+str(test_acc_last)

                            #print len(picks)
                            #time.sleep(1)
                            data_slice_x=data[picks,col_step].reshape([1,-1])

                            data_slice_xp=np.arange(0,data_slice_x.shape[1]).reshape([1,-1])
                            data_slice_y=data[picks,e].reshape([1,-1])
                            plt.plot(data_slice_xp.T, data_slice_y.T,linestyle=lin,color=col,label=lab)

                            plt.xlim(0,1000*20.)
                            plt.ylabel(err_col_lab[edx-1])
                            if edx==2:
                                plt.xlabel('Step')

                            if edx==2:
                                legend = plt.legend(loc=(0,-1.),labelspacing=0)
        plt.show()

