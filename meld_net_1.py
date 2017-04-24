import numpy as np
import numpy.matlib as matlib 
import tensorflow as tf

def mats_4_err_calc(locate):
    w0 = matlib.repmat(np.identity(3),1,locate)
    print 'w0', w0.shape
    w1 = np.zeros((3*locate,locate))
    for l in range(0,locate):
        w1[3*l:3*(l+1),l]=1./3.
    print 'w1', w1.shape
    return w0,w1

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
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('sttdev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram('histogram/'+name, var)

def sph2cart(r,th,ph):
    
    x = tf.multiply(tf.multiply(r,tf.cos(th)),tf.cos(ph))
    y = tf.multiply(tf.multiply(r,tf.cos(th)),tf.sin(ph))
    z = tf.multiply(r,tf.sin(th))

    return x,y,z

class meld:
    def __init__(self,learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps=None,n_lstm=None,n_layer=None, cnn=True, rnn=True, locate=True):
        self.learning_rate=learning_rate
        print "learning rate: ", self.learning_rate
        self.meas_dims=meas_dims
        print "meas_dims: ", self.meas_dims
        self.k_conv=k_conv
        print "kconv: ", self.k_conv
        self.k_pool=k_pool
        print "k_pool: ", self.k_pool
        self.n_chan_in=n_chan_in
        print "n_chan_in: ", self.n_chan_in
        self.n_conv1=n_conv1
        print "n_conv1: ", self.n_conv1
        self.n_conv2=n_conv2
        print "n_conv2: ", self.n_conv2
        self.n_out=n_out
        self.n_obs=n_out
        print "n_out: ", self.n_out        
        self.n_steps=n_steps
        print "n_steps: ", self.n_steps
        self.n_lstm=n_lstm
        print "n_lstm: ", self.n_lstm
        self.n_layer=n_layer
        print "n_layer: ", self.n_layer
        self.cnn=cnn
        print "CNN: ",self.cnn
        self.rnn=rnn
        print "RNN: ",self.rnn
        if self.rnn is True and self.cnn is 'fft':
            self.rnn = False
            print "RNN needs to be: ",self.rnn," when using fft option"
        self.locate=locate
        print "Locate: ",self.locate
        if self.locate is True:
            self.locate=1
        if self.locate is not False:
            print "For nearest-neighbor technique, n_out must be 3"
            self.n_out=3
            print "n_out: ", self.n_out
        if self.cnn is not False:
            self.n_dense=int((self.meas_dims[0]-self.k_conv+1)/self.k_pool-self.k_conv+1)*int((self.meas_dims[1]-self.k_conv+1)/self.k_pool-self.k_conv+1)*self.n_conv2
        else:
            self.n_dense=self.meas_dims
        print "n_dense: ", self.n_dense
        

        if self.locate is not False:
            self.w0, self.w1 = mats_4_err_calc(locate)
    def cnn_nn(self):        
        # Store layers weight & bias
        with tf.name_scope('convolutional_layer_1'):
            wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
            bc1 = tf.Variable(tf.random_normal([self.n_conv1]))
            variable_summaries(wc1, 'wc1')
            variable_summaries(bc1, 'bc1')
            self.measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
            conv1 = conv2d(self.measfold,wc1,bc1,name="conv1")
            conv1 = max_pool(conv1,k=self.k_pool)
            # Apply Dropout
            conv1 = tf.nn.dropout(conv1,self.dropoutPH)

        with tf.name_scope('convolutional_layer_2'):
            wc2 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_conv1, self.n_conv2])) # kxk conv, 2 input, n_conv outputs
            bc2 = tf.Variable(tf.random_normal([self.n_conv2]))
            variable_summaries(wc2, 'wc2')
            variable_summaries(bc2, 'bc2')
            conv2 = conv2d(conv1,wc2,bc2,name="conv2" )
            # Apply Dropout
            conv2 = tf.nn.dropout(conv2, self.dropoutPH)

        with tf.name_scope('dense_layer'):
            dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_out],stddev=100./self.n_dense))
            bd = tf.Variable(tf.random_normal([self.n_out],stddev=100./self.n_dense))
            variable_summaries(wd, 'wd')
            variable_summaries(bd, 'bd')
            dense_out = tf.add(tf.matmul(dense, wd),bd)
            #dense_out = tf.matmul(dense, wd)
            self.logits = tf.nn.dropout(dense_out, self.dropoutPH)
            self.logits_last = tf.nn.dropout(dense_out, self.dropoutPH)
            if self.locate is not False:
                self.qhat = self.logits
                self.qhat_last = self.logits
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = self.logits
            self.reg=tf.multiply(self.betaPH,
                                 tf.add(tf.add(tf.nn.l2_loss(wd),tf.nn.l2_loss(wc1)),
                                        tf.nn.l2_loss(wc2)))
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits,1)

    def rnn_nn(self):
        with tf.name_scope('dense_layer'):
            dense = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm]))
            bd = tf.Variable(tf.random_normal([self.n_lstm]))
            variable_summaries(wd, 'wd')
            variable_summaries(bd, 'bd')
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")#try relu
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
        with tf.name_scope('rnn_layer'):
            #now predict sequence of firing
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_lstm,state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layer,state_is_tuple=True)  
            
            data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

            val = tf.transpose(output,[1,0,2])#nxbxn_lstm
            last = val[-1,:,:]#tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
            outs = tf.reshape(output, [-1, self.n_lstm])#n*bxn_lstm
                
            wrnn = tf.Variable(tf.random_normal([self.n_lstm,self.n_out],stddev=100./self.n_lstm))
            brnn = tf.Variable(tf.random_normal([self.n_out],stddev=100./self.n_lstm))
            variable_summaries(wrnn, 'wrnn')
            variable_summaries(brnn, 'brnn')

            self.logits = tf.add(tf.matmul(outs,wrnn),brnn)#logits - b*nxp             
            self.logits_last=tf.add(tf.matmul(last,wrnn),brnn)#logits - bxp

            if self.locate is not False:
                self.qhat = self.logits
                self.qhat_last = self.logits_last                
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")
            
            self.reg=tf.multiply(self.betaPH,tf.nn.l2_loss(wd))
            
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)
        
    def cnn_rnn(self):
        # Store layers weight & bias
        with tf.name_scope('convolutional_layer_1'):
            wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
            bc1 = tf.Variable(tf.random_normal([self.n_conv1]))
            variable_summaries(wc1, 'wc1')
            variable_summaries(bc1, 'bc1')
            #reshape to fold batch and timestep indices, needed for compatibility with conv2d
            self.measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
            conv1 = conv2d(self.measfold,wc1,bc1,name="conv1")
            conv1 = max_pool(conv1,k=self.k_pool)
            # Apply Dropout
            conv1 = tf.nn.dropout(conv1,self.dropoutPH)

        with tf.name_scope('convolutional_layer_2'):
            wc2 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_conv1, self.n_conv2])) # kxk conv, 2 input, n_conv outputs
            bc2 = tf.Variable(tf.random_normal([self.n_conv2]))
            variable_summaries(wc2, 'wc2')
            variable_summaries(bc2, 'bc2')
            conv2 = conv2d(conv1,wc2,bc2,name="conv2" )
            # Apply Dropout
            conv2 = tf.nn.dropout(conv2, self.dropoutPH)

        with tf.name_scope('dense_layer'):
            dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm]))
            bd = tf.Variable(tf.random_normal([self.n_lstm]))
            variable_summaries(wd, 'wd')
            variable_summaries(bd, 'bd')
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
        with tf.name_scope('rnn_layer'):
            #now predict sequence of firing
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_lstm,state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layer,state_is_tuple=True)  

            data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

            val = tf.transpose(output,[1,0,2])#nxbxn_lstm
            last = val[-1,:,:]#tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
            outs = tf.reshape(output, [-1, self.n_lstm])#n*bxn_lstm
                
            wrnn = tf.Variable(tf.random_normal([self.n_lstm,self.n_out],stddev=100./self.n_lstm))
            brnn = tf.Variable(tf.random_normal([self.n_out],stddev=100./self.n_lstm))
            variable_summaries(wrnn, 'wrnn')
            variable_summaries(brnn, 'brnn')

            self.logits = tf.add(tf.matmul(outs,wrnn),brnn)#self.logits - b*nxp
            self.logits_last=tf.add(tf.matmul(last,wrnn),brnn)#logits - bxp

            if self.locate is not False:
                self.qhat = self.logits
                self.qhat_last = self.logits_last                
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")

            self.reg=tf.multiply(self.betaPH,
                            tf.add(tf.add(tf.nn.l2_loss(wd),tf.nn.l2_loss(wc1)),
                                   tf.nn.l2_loss(wc2)))
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)

    def rnn_nn2(self):
        with tf.name_scope('dense_layer'):
            dense = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm[0]]))
            bd = tf.Variable(tf.random_normal([self.n_lstm[0]]))
            variable_summaries(wd, 'wd')
            variable_summaries(bd, 'bd')
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")#try relu
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
                
        with tf.name_scope('encoder_layer'):
            #now predict sequence of firing
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_lstm[0],state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layer[0],state_is_tuple=True)  
            
            data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm[0]])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
                
        with tf.name_scope('between_layer'):
            wb = tf.Variable(tf.random_normal([self.n_lstm[0],self.n_lstm[1]]))
            bb = tf.Variable(tf.random_normal([self.n_lstm[1]]))
            variable_summaries(wb, 'wb')
            variable_summaries(bb, 'bb')

            output1 = tf.reshape(tf.nn.relu(tf.add(tf.matmul(tf.reshape(output,[-1,self.n_lstm[0]]),wb),bb)),
                                 [-1, self.n_steps, self.n_lstm[1]])           
                               
        with tf.name_scope('decoder_layer'):
            #now predict sequence of firing
            cell2 = tf.contrib.rnn.BasicLSTMCell(self.n_lstm[1],state_is_tuple=True)
            cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=self.dropoutPH)
            cell2 = tf.contrib.rnn.MultiRNNCell([cell2] * self.n_layer[1],state_is_tuple=True)  
            
            output2, state2 = tf.nn.dynamic_rnn(cell2, output1, dtype=tf.float32)

            val = tf.transpose(output2,[1,0,2])#nxbxn_lstm
            last = val[-1,:,:]#tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
            outs = tf.reshape(output2, [-1, self.n_lstm[1]])#n*bxn_lstm
                
            wrnn = tf.Variable(tf.random_normal([self.n_lstm[1],self.n_out],stddev=100./self.n_lstm[1]))
            brnn = tf.Variable(tf.random_normal([self.n_out],stddev=100./self.n_lstm[1]))
            variable_summaries(wrnn, 'wrnn')
            variable_summaries(brnn, 'brnn')

            self.logits = tf.add(tf.matmul(outs,wrnn),brnn)#logits - b*nxp             
            self.logits_last=tf.add(tf.matmul(last,wrnn),brnn)#logits - bxp

            if self.locate is not False:
                self.qhat = self.logits
                self.qhat_last = self.logits_last                
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")
            
            self.reg=tf.multiply(self.betaPH,tf.nn.l2_loss(wd))
            
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)
        
    def cnn_rnn2(self):
        # Store layers weight & bias
        with tf.name_scope('convolutional_layer_1'):
            wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
            bc1 = tf.Variable(tf.random_normal([self.n_conv1]))
            variable_summaries(wc1, 'wc1')
            variable_summaries(bc1, 'bc1')
            #reshape to fold batch and timestep indices, needed for compatibility with conv2d
            self.measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
            conv1 = conv2d(self.measfold,wc1,bc1,name="conv1")
            conv1 = max_pool(conv1,k=self.k_pool)
            # Apply Dropout
            conv1 = tf.nn.dropout(conv1,self.dropoutPH)

        with tf.name_scope('convolutional_layer_2'):
            wc2 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_conv1, self.n_conv2])) # kxk conv, 2 input, n_conv outputs
            bc2 = tf.Variable(tf.random_normal([self.n_conv2]))
            variable_summaries(wc2, 'wc2')
            variable_summaries(bc2, 'bc2')
            conv2 = conv2d(conv1,wc2,bc2,name="conv2" )
            # Apply Dropout
            conv2 = tf.nn.dropout(conv2, self.dropoutPH)
        with tf.name_scope('dense_layer'):
            dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv2 output to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm[0]]))
            bd = tf.Variable(tf.random_normal([self.n_lstm[0]]))
            variable_summaries(wd, 'wd')
            variable_summaries(bd, 'bd')
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")#try relu
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
                
        with tf.name_scope('encoder_layer'):
            #now predict sequence of firing
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_lstm[0],state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layer[0],state_is_tuple=True)  
            
            data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm[0]])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
                
        with tf.name_scope('between_layer'):
            wb = tf.Variable(tf.random_normal([self.n_lstm[0],self.n_lstm[1]]))
            bb = tf.Variable(tf.random_normal([self.n_lstm[1]]))
            variable_summaries(wb, 'wb')
            variable_summaries(bb, 'bb')

            output1 = tf.reshape(tf.nn.relu(tf.add(tf.matmul(tf.reshape(output,[-1,self.n_lstm[0]]),wb),bb)),
                                 [-1, self.n_steps, self.n_lstm[1]])           
                
        with tf.name_scope('decoder_layer'):
            #now predict sequence of firing
            cell2 = tf.contrib.rnn.BasicLSTMCell(self.n_lstm[1],state_is_tuple=True)
            cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=self.dropoutPH)
            cell2 = tf.contrib.rnn.MultiRNNCell([cell2] * self.n_layer[1],state_is_tuple=True)  
            
            output2, state2 = tf.nn.dynamic_rnn(cell2, output1, dtype=tf.float32)

            val = tf.transpose(output2,[1,0,2])#nxbxn_lstm
            last = val[-1,:,:]#tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
            outs = tf.reshape(output2, [-1, self.n_lstm[1]])#n*bxn_lstm
                
            wrnn = tf.Variable(tf.random_normal([self.n_lstm[1],self.n_out],stddev=100./self.n_lstm[1]))
            brnn = tf.Variable(tf.random_normal([self.n_out],stddev=100./self.n_lstm[1]))
            variable_summaries(wrnn, 'wrnn')
            variable_summaries(brnn, 'brnn')

            self.logits = tf.add(tf.matmul(outs,wrnn),brnn)#logits - b*nxp             
            self.logits_last=tf.add(tf.matmul(last,wrnn),brnn)#logits - bxp

            if self.locate is not False:
                self.qhat = self.logits
                self.qhat_last = self.logits_last                
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")
            
            self.reg=tf.multiply(self.betaPH,tf.nn.l2_loss(wd))
            
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)


    def mlp(self):
        with tf.name_scope('input_layer'):
            dense = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm]))
            bd = tf.Variable(tf.random_normal([self.n_lstm]))
            variable_summaries(wd, 'wd')
            variable_summaries(bd, 'bd')
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")#try relu
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
       
        with tf.name_scope('output_layer'):
            wo = tf.Variable(tf.random_normal([self.n_lstm, self.n_out],stddev=100./self.n_lstm))
            bo = tf.Variable(tf.random_normal([self.n_out],stddev=100./self.n_lstm))
            variable_summaries(wo, 'wo')
            variable_summaries(bo, 'bo')
            self.logits = tf.add(tf.matmul(dense_out, wo),bo)
            #self.logits = tf.matmul(dense_out, wo)
            if self.locate is not False:
                self.qhat = self.logits
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
            
            self.reg=tf.add(tf.multiply(self.betaPH,tf.nn.l2_loss(wd)),tf.multiply(self.betaPH,tf.nn.l2_loss(wo)))
            
            self.A=tf.argmax(self.logits,1)

    def network(self):
        self.dropoutPH = tf.placeholder(tf.float32, name="dropout")
        self.betaPH =  tf.placeholder(tf.float32, name="beta")
        if self.cnn is False:
            self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims), name="meas")
        elif self.cnn is 'fft':
            self.measPH=tf.placeholder(tf.float32,shape=(None, self.meas_dims[0], self.meas_dims[1],self.n_chan_in), name="meas")
        else:
            self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims[0], self.meas_dims[1],self.n_chan_in), name="meas")

        if self.rnn is True:
            if type(self.n_lstm) is not int:                
                if self.cnn is not False:
                    self.cnn_rnn2()
                else:
                    self.rnn_nn2()
            else:
                if self.cnn is not False:
                    self.cnn_rnn()
                else:
                    self.rnn_nn()
        else:            
            if self.cnn is not False:
                self.cnn_nn()
            else:
                self.mlp()          

    def trainer(self):
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.saver = tf.train.Saver()
            
    def initializer(self):
        self.init_step = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()
    def cost(self):
        if self.locate is not False:
            self.W0 = tf.constant(self.w0, dtype=tf.float32)
            self.W1 = tf.constant(self.w1, dtype=tf.float32)
        
        self.qtrainPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.n_obs), name="qtrain")
        self.qtrain_unflat = tf.reshape(self.qtrainPH,[-1,self.n_obs])#b*nxp
        with tf.name_scope('cost'):
            if ((self.rnn is False) and (self.cnn is not 'fft')):                
                B=tf.argmax(self.qtrain_unflat,1)
                qtrain_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation

                self.cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=qtrain_OH),name="cross"),self.reg)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.A,B),tf.float32),name="accuracy")
               
                if self.locate is not False:
                    with tf.name_scope('rep_outputs'):
                        qhat_rep = tf.matmul(self.qhat,self.W0)#b*nx3x3x3l
                        SE = tf.matmul(tf.square(tf.subtract(qhat_rep,self.qtrain_unflat)),self.W1)#b*nx3lx3lxl
                        SEbn = tf.reduce_min(SE,1)#b*n
                        SSE = tf.reduce_mean(SEbn)#1
                        self.rmse = tf.add(tf.sqrt(SSE,name="rmse"),self.reg)
                else:
                    self.rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.qhat,self.qtrain_unflat))),name="rmse"),self.reg)

                if self.locate is not False:
                    self.cost = self.rmse 
                else:
                    self.cost = self.cross

            else:#yes rnn or cnn is fft
                B=tf.argmax(self.qtrain_unflat,1)#b*nx1
                qtrain_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
        
                qtrain_tran = tf.transpose(self.qtrainPH,[1,0,2])#nxbxp
                self.qtrain_last = tf.gather(qtrain_tran, int(qtrain_tran.get_shape()[0])-1)#bxp
                BB=tf.argmax(self.qtrain_last,1)#bx1
                self.qtrain_last_OH = tf.one_hot(BB,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
                
                self.cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_last, labels=self.qtrain_last_OH),name="cross_last"),self.reg)
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(self.AA,BB),tf.float32),name="accuracy_last")
                if self.locate is not False:
                    with tf.name_scope('rep_outputs'):
                        qhat_rep = tf.matmul(self.qhat_last,self.W0)#bx3x3x3l
                        SE = tf.matmul(tf.square(tf.subtract(qhat_rep,self.qtrain_last)),self.W1)#bx3lx3lxl
                        SEnb = tf.reduce_min(SE,axis=1)#b
                        SSE = tf.reduce_mean(SEnb)#1
                        self.rmse_last = tf.add(tf.sqrt(SSE,name="rmse"),self.reg)
                else:
                    self.rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.qtrain_last,self.qhat_last))),name="rmse_last"),self.reg)
                
                if self.locate is not False:
                    self.cost = self.rmse_last 
                else:
                    self.cost = self.cross_last

        with tf.name_scope('summaries'):
            self.train_summary = tf.summary.scalar('mean/train_cost', self.cost)
            self.valid_summary = tf.summary.scalar('mean/valid_cost', self.cost)
            if self.locate is False:
                if self.rnn is True or self.cnn is 'fft':
                    self.train_acc_summary = tf.summary.scalar('mean/train_accuracy', self.accuracy_last)
                    self.valid_acc_summary = tf.summary.scalar('mean/valid_accuracy', self.accuracy_last)            
                else:
                    self.train_acc_summary = tf.summary.scalar('mean/train_accuracy', self.accuracy)
                    self.valid_acc_summary = tf.summary.scalar('mean/valid_accuracy', self.accuracy)
