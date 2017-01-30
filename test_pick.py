import numpy as np
import time 
n_tot = 879
n_batch =100
n_test = 50
n_val = 50
a = np.arange(0,n_tot)
test = np.random.choice(a,n_test,replace=False)
print test
time.sleep(1)

p = np.ones(a.shape)/(n_tot-n_test)
p[test]=0.
val = np.random.choice(a,n_val,replace=False,p=p)
print val
time.sleep(1)

p[val]=0.
p=p*(n_tot-n_test)/(n_tot-n_test-n_val)
batch=np.random.choice(a,n_batch,replace=False,p=p)
print batch
time.sleep(1)

print np.intersect1d(batch,val)
print np.intersect1d(batch,test)
