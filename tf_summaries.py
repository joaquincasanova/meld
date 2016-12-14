import numpy as np
from numpy import matlib
import sphere
import dipole_class
import tensorflow as tf

# Create model
def conv2d(img, w, b, name):
    return tf.nn.relu(tf.nn.bias_add\
                      (tf.nn.conv2d(img, w,\
                                    strides=[1, 1, 1, 1],\
                                    padding='VALID'),b),name=name)

def max_pool(img, k):
    return tf.nn.max_pool(img, \
                          ksize=[1, k, k, 1],\
                          strides=[1, k, k, 1],\
                          padding='VALID')

def variable_summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

# Parameters
learning_rate = .005
batches=10
per_batch=int(1./learning_rate)
batch_size=1000/batches
test_size=batch_size
Ts = 1#s, sample time
n_steps=1#number of time steps
#print per_batch
meas_dims=[11,11]
dipole_dims=[4,5,5]
C=-np.float32(10)/np.log(0.99)
dropout = np.exp(-batches/C) # Dropout, probability to keep units
print dropout

instance=dipole_class.dipole(Ts, n_steps,meas_dims,dipole_dims,batch_size)#,orient=True
instance.batch_sequence_gen()
meas_img=instance.meas_img
m=instance.m
p=instance.p
qtrue=instance.qtrue
#instance.fields_plot()

k_pool=1
k_conv=3
n_out=p
n_in=meas_dims[0]*meas_dims[1]*2
n_conv1=3
n_conv2=5
n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2

dropoutPH = tf.placeholder(tf.float32, name="dropout")
measPH=tf.placeholder(tf.float32,shape=(None,meas_dims[0],meas_dims[1],2), name="meas")
qtruePH=tf.placeholder(tf.float32,shape=(None, p), name="qtrue")

with tf.name_scope('convolutional_layer_1'):
    wc1 = tf.Variable(tf.random_normal([k_conv, k_conv, 2, n_conv1])) # kxk conv, 2 input, n_conv outputs
    bc1 = tf.Variable(tf.random_normal([n_conv1]))
    conv1 = conv2d(meas_img,wc1,bc1,name="conv1")
    conv1 = max_pool(conv1,k=k_pool)
    conv1 = tf.nn.dropout(conv1,dropoutPH)
    variable_summaries(wc1, "wc1")
    variable_summaries(bc1, "bc1")
    variable_summaries(conv1, "conv1")

with tf.name_scope('convolutional_layer_2'):
    wc2 = tf.Variable(tf.random_normal([k_conv, k_conv, n_conv1, n_conv2])) # kxk conv, 2 input, n_conv outputs
    bc2 = tf.Variable(tf.random_normal([n_conv2]))
    conv2 = conv2d(conv1,wc2,bc2,name="conv2")
    conv2 = tf.nn.dropout(conv2,dropoutPH)
    
    variable_summaries(wc2, "wc2")
    variable_summaries(bc2, "bc2")
    variable_summaries(conv2, "conv2")

with tf.name_scope('dense_layer'):
    wd = tf.Variable(tf.random_normal([n_dense, n_out])) # fully connected, image inputs, n_out outputs
    bd = tf.Variable(tf.random_normal([n_out]))
    dense = tf.reshape(conv2, [-1, n_dense]) # Reshape conv2 output to fit dense layer input
    logits = tf.add(tf.matmul(dense,wd),bd)#logits
    qhat = tf.nn.softmax(logits,name="qhat") 
    
    variable_summaries(wd, "wd")
    variable_summaries(bd, "bd")
    variable_summaries(dense, "dense")
    variable_summaries(qhat, "qhat")

with tf.name_scope('cost'):
    A=tf.argmax(logits,1)
    B=tf.argmax(qtruePH,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qhat,qtruePH))),name="rmse")
    variable_summaries(rmse, "rmse")
    cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, qtruePH),name="cross")
    variable_summaries(cross, "cross")
    variable_summaries(accuracy, "accuracy")

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)

init_step = tf.initialize_all_variables()
merged = tf.merge_all_summaries()
  
with tf.Session() as session:

    if tf.gfile.Exists("/tmp/tensorflowlogs"):
        tf.gfile.DeleteRecursively("/tmp/tensorflowlogs")
    tf.gfile.MakeDirs("/tmp/tensorflowlogs")
    train_writer = tf.train.SummaryWriter("/tmp/tensorflowlogs",session.graph)
    session.run(init_step)
    for step in range(0,per_batch*batches):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        summary, _ , guess,ce,acc,err  = session.run([merged, train_step, qhat, cross, accuracy, rmse],
                                                 feed_dict={qtruePH: qtrue, measPH: meas_img, dropoutPH: dropout},
                                                 options=run_options,
                                                 run_metadata=run_metadata)
        
        print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ",err
        if step % per_batch ==0 and step!=0:#generate a new batch
            #instance.dipole_plot_scalar(guess)

            instance=dipole_class.dipole(Ts, n_steps,meas_dims,dipole_dims,batch_size)#,orient=True
            instance.batch_sequence_gen()
            meas_img=instance.meas_img
            qtrue=instance.qtrue

        train_writer.add_run_metadata(run_metadata, 'train_step%03d' % step)
        train_writer.add_summary(summary, step)

    instance=dipole_class.dipole(Ts, n_steps,meas_dims,dipole_dims,test_size)#,orient=True
    instance.batch_sequence_gen()
    meas_img=instance.meas_img
    qtrue=instance.qtrue
    #print qtrue.shape
    #print meas_img.shape
    train_writer.add_run_metadata(run_metadata, 'test_step%03d' % step)
    train_writer.add_summary(summary, step)

    guess, ce, acc, err  = session.run([qhat, cross, accuracy, rmse],
                                       feed_dict={qtruePH: qtrue, measPH: meas_img, dropoutPH: dropout},
                                       options=run_options,
                                       run_metadata=run_metadata)
    print "Test Step: CE: ",ce, " Accuracy: ", acc, "RMSE: ",err
    #instance.dipole_plot_scalar(guess)
    train_writer.close()
