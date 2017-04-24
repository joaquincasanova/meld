import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']
#['zss',True, False],['zss',True, True],['zss',False, False],['zss',False, True],
pca=True
rand_test=True
for subject_id in ['aud',7]:
    idx = 0
    for locate in [1]:
        if locate  is False:
            subsample=1
        for cnn in [True,False]:
            for rnn in [True,False]:
                    if subject_id is 'aud':
                        treats=[None]#,'left/auditory', 'right/auditory', 'left/visual', 'right/visual']
                    elif subject_id is 'rat':
                        treats=[None]
                    else:
                        treats=[None]#,'face/famous','scrambled','face/unfamiliar']

                    for treat in treats:
                        if treat is not None:
                            lab_treat=treat.replace("/","_")
                        else:
                            lab_treat='None'


                        fieldnames=['batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','cost']
                        name='/home/jcasa/meld/tnrse2017/tf1_steps_subject_%s_pca_all_%s_rand_%s_cnn_%s_rnn_%s_locate_%s_treat_%s' % (subject_id, pca, rand_test, cnn, rnn,locate,lab_treat)
                        fname = name + '.csv' 

                        data=np.zeros([1,10])
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
                                    col_k_conv=header.index('k_conv')
                                    col_n_conv1=header.index('n_conv1')
                                    col_n_conv2=header.index('n_conv2')
                                    col_n_layer=header.index('n_layer')
                                    col_n_lstm=header.index('n_lstm')
                                    col_n_steps=header.index('n_steps')

                                    col_step=header.index('train step')
                                    col_cost=header.index("cost")

                                    col_batches=header.index("batches")
                                    col_size=header.index("batch_size")
                                    col_rate=header.index("learning rate")
                                    col_per=header.index("per_batch")
                                    col_drop=header.index("dropout")
                                    col_beta=header.index("beta")
                                else:
                                    try: 
                                        this=np.array([np.float(row[col_step]),np.float(row[col_batches]),np.float(row[col_per]),
                                                       np.float(row[col_size]),np.float(row[col_k_conv]),np.float(row[col_n_conv1]),
                                                       np.float(row[col_n_conv2]),np.float(row[col_n_layer]),np.float(row[col_n_lstm]),
                                                       np.float(row[col_cost])])
                                        this=this.reshape([10,1]).T
                                        data=np.append(data,this,axis=0)
                                    except:
                                        continue
                                rownum+=1

                        finally:
                            csvfile.close()

                    col_step=0
                    col_per=1
                    col_batches=2
                    col_size=3
                    col_k_conv=4
                    col_n_conv1=5
                    col_n_conv2=6
                    col_n_layer=7
                    col_n_lstm=8
                    col_cost=9

                    params_list = [[3,3,5,10,2,.2,.1,.1]]

                    err_col=9

                    err_col_lab=['RMSE Train (mm)','RMSE Validation (mm)']

                    data=np.delete(data,(0),axis=0)
                    leg=[('-','b'), ('-','g'),('-', 'r'), ('-','k'),(':','b'), (':','g'),(':', 'r'), (':','k'),('--','b'), ('--','g'),('--', 'r'), ('--','k'),('-.','b'), ('-.','g'),('-.', 'r'), ('-.','k')]
                    for [kn, cn1, cn2, ls, la, ts, vs, ns] in params_list:
                        idx+=1 


                        col=leg[idx-1][1]
                        lin=leg[idx-1][0]

                        lab='CNN '+str(cnn)+', RNN '+str(rnn)
                        print lab

                        picks = np.where(data[:,col_n_lstm]==ls)
                        print picks
                        picks = np.intersect1d(picks, np.where(data[:,col_k_conv]==kn))
                        print picks
                        picks = np.intersect1d(picks, np.where(data[:,col_n_conv1]==cn1))
                        print picks
                        picks = np.intersect1d(picks, np.where(data[:,col_n_conv2]==cn2))
                        print picks
                        picks = np.intersect1d(picks, np.where(data[:,col_n_layer]==la))
                        print picks
                        test = picks
                        val = picks
                        picks = np.intersect1d(picks, np.where(data[:,col_step]>=0))
                        test = np.intersect1d(test, np.where(data[:,col_step]<=-2))

                        test_last=np.mean(data[test,err_col])
                        val = np.intersect1d(val, np.where(data[:,col_step]==-1))

                        #print data[val,err_col]

                        val_last=np.mean(data[val,err_col])

                        lab=lab+', test cost='+str(test_last)

                        #print len(picks)
                        #time.sleep(1)
                        data_slice_x=np.arange(0,data[picks,col_step].reshape([1,-1]).shape[1]).reshape([1,-1])
                        data_slice_y=data[picks,err_col].reshape([1,-1])

                        data_val_x=(np.arange(0,data[val,err_col].shape[0]).reshape([1,-1])+1)*100
                        data_val_y=data[val,err_col].reshape([1,-1])

                        plt.subplot(2, 1, 1)

                        plt.semilogy(data_slice_x.T, data_slice_y.T,linestyle=lin,color=col,label=lab)
                        if subject_id is 'aud':
                            title = 'Subject: '+subject_id
                        else:
                            title = 'Subject: faces #'+str(subject_id)
                        plt.title(title)
                        plt.xlim(0,500*10.)
                        plt.ylim(0,100.)
                        plt.ylabel(err_col_lab[0])

                        plt.subplot(2, 1, 2)

                        plt.semilogy(data_val_x.T, data_val_y.T,linestyle=lin,color=col,label=lab)

                        plt.xlim(0,500*10.)
                        plt.ylim(0,100.)
                        plt.ylabel(err_col_lab[1])

                        plt.xlabel('Step')

    legend = plt.legend(loc='best',labelspacing=0)
    plt.show()

