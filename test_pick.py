import numpy as np
import time 
n_tot = 879
n_test = int(.1*n_tot)
n_val =  int(.1*(n_tot-n_test))
n_batch = int(.1*(n_tot-n_test))
print n_tot, n_test, n_val, n_batch
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
p=p*float(n_tot-n_test)/float(n_tot-n_test-n_val)
batches=int((n_tot-n_test-n_val)/n_batch)
print batches
time.sleep(1)

for bat in range(0,batches):
    batch=np.random.choice(a,n_batch,replace=False,p=p)
    print bat, batch
    time.sleep(1)
    p[batch]=0
    p=p*float(n_tot-n_test-n_val-n_batch*bat)/float(n_tot-n_test-n_val-n_batch*(bat+1))

assert np.intersect1d(batch,val).size is 0
assert np.intersect1d(batch,test).size is 0
