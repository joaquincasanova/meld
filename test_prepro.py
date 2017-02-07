import nn_prepro
import time
import numpy as np
test = np.random.choice(879,10,replace=False)
test_id=7
pca=True
subsample=1
meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(test_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=True,locate=True)
