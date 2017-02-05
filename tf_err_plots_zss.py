import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']
for rand_test in [False, True]:
    for train_id in [7]:
        for test_id in [7]:
        fname = './data/nn_real_zss_%s_%s_%s.csv' % (train_id, test_id, rand_test)

        data=np.zeros([1,13])
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
                        this=np.array([np.float(row[col_step]),np.float(row[col_per]),np.float(row[col_size]),np.float(row[col_n_conv1]), np.float(row[col_n_conv2]),np.float(row[col_n_layer]), np.float(row[col_n_lstm]),  np.float(row[col_rmse]),np.float(row[col_acc]), np.float(row[col_ce]), np.float(row[col_rmse_last]),np.float(row[col_acc_last]), np.float(row[col_ce_last])])
                        this=this.reshape([13,1]).T
                        data=np.append(data,this,axis=0)
                    except:
                        continue
                rownum+=1

        finally:
            csvfile.close()

        col_step=0
        col_per=1
        col_size=2
        col_n_conv1=3
        col_n_conv2=4
        col_n_layer=5
        col_n_lstm=6
        col_rmse=7         
        col_acc=8       
        col_ce=9        
        col_rmse_last=10        
        col_acc_last=11       
        col_ce_last=12   

        params_list = [[2,5,10,2,175,140,140],[2,5,10,2,175,70,70]]

        err_col_vals=[7,10]#[3,4,5,6,7,8]

        err_col_lab=['RMSE','RMSE Last']#['RMSE','Accuracy','Cross Entropy','RMSE Last','Accuracy Last','Cross Entropy Last']

        data=np.delete(data,(0),axis=0)
        leg=[('-','b'), ('-','g'),('-', 'r'), ('-','k'),(':','b'), (':','g'),(':', 'r'), (':','k'),('--','b'), ('--','g'),('--', 'r'), ('--','k'),('-.','b'), ('-.','g'),('-.', 'r'), ('-.','k')]
        idx = 0
        for [cn1, cn2, ls, la, ts, vs, ns] in params_list:
            idx+=1
            edx=0
            for e in err_col_vals:
                edx+=1 
                
                plt.subplot(3, 1, edx)
                
                col=leg[idx-1][1]
                lin=leg[idx-1][0]

                lab='n_conv1='+str(cn1)+', n_conv2='+str(cn2)+', n_lstm='+str(ls)+', n_layers='+str(la)+', batch size='+str(ns)
                print lab
                picks = np.where(data[:,col_n_lstm]==ls)
                print picks
                picks = np.intersect1d(picks, np.where(data[:,col_n_conv1]==cn1))
                print picks
                picks = np.intersect1d(picks, np.where(data[:,col_n_conv2]==cn2))
                print picks
                picks = np.intersect1d(picks, np.where(data[:,col_n_layer]==la))
                print picks
                picks = np.intersect1d(picks, np.where(data[:,col_size]==ns))
                print picks
                test = picks
                val = picks
                picks = np.intersect1d(picks, np.where(data[:,col_step]>=0))
                test = np.intersect1d(test, np.where(data[:,col_step]<=-2))
                test_acc_last=np.mean(data[test,10])
                val = np.intersect1d(val, np.where(data[:,col_step]==-1))
                print data[val,10]
                val_acc_last=np.mean(data[val,10])
                lab=lab+', val rmse='+str(val_acc_last)+', test rmse='+str(test_acc_last)

                #print len(picks)
                #time.sleep(1)
                data_slice_x=data[picks,col_step].reshape([1,-1])
                
                data_slice_xp=np.arange(0,data_slice_x.shape[1]).reshape([1,-1])
                data_slice_y=data[picks,e].reshape([1,-1])
                plt.plot(data_slice_xp.T, data_slice_y.T,linestyle=lin,color=col,label=lab)

                plt.xlim(0,500*10.)
                plt.ylabel(err_col_lab[edx-1])
                if edx==2:
                    plt.xlabel('Step')

                if edx==2:
                    legend = plt.legend(loc=(0,-1.),labelspacing=0)
        plt.show()

