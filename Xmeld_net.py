import numpy as np
import numpy.matlib as matlib 
import tensorflow as tf

def conv_relu(img, kernel_shape, bias_shape, dropout, beta):
    with tf.variable_scope("conv_relu"):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.random_normal_initializer())

        conv = tf.nn.conv2d(img, weights,
            strides=[1, 1, 1, 1], padding='VALID')

        variable_summaries(weights, "weights")
        variable_summaries(biases, "biases")
        reg=tf.multiply(beta,tf.nn.l2_loss(weights))
    return tf.nn.dropout(tf.nn.relu(conv + biases),dropout),reg

def cnn_2layers(img, k_conv, n_chan_in, n_conv_1, n_conv_2, dropout,beta):        
    # Store layers weight & bias

    kernel_shape_1 = [k_conv, k_conv, n_chan_in, n_conv_1]
    kernel_shape_2 = [k_conv, k_conv, n_conv_1, n_conv_2]
    bias_shape_1 = [n_conv_1]
    bias_shape_2 = [n_conv_2]
    with tf.variable_scope('convolutional_layer_1'):
        out1,reg1 = conv_relu(img, kernel_shape_1, bias_shape_1, dropout,beta)
    with tf.variable_scope('convolutional_layer_2'):
        out2,reg2 = conv_relu(out1, kernel_shape_2, bias_shape_2, dropout,beta)
    return out2, reg1+reg2

def rnn(dense_in,n_lstm,n_steps,n_layer,dropout,beta):                        
    with tf.variable_scope('rnn'):
        #now predict sequence of firing
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_lstm,state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layer,state_is_tuple=True)  
            
        data = tf.reshape(dense_in, [-1, n_steps, n_lstm])
        output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        
        last = output[:,-1,:]        
        outs = tf.reshape(output, [-1, n_lstm])#n*bxn_lstm
    return last, outs

def dense(dense_in, n_dense_in, n_dense_out,dropout,beta,std_dense):
    with tf.variable_scope('dense_layer'):
        wd = tf.get_variable("weights",[n_dense_in, n_dense_out],initializer=tf.random_normal_initializer(0.0,std_dense))
        bd = tf.get_variable("biases",[n_dense_out],initializer=tf.random_normal_initializer(0.0,std_dense))
        dense_out = tf.nn.relu(tf.add(tf.matmul(dense_in, wd),bd))
        dense_out = tf.nn.dropout(dense_out, dropout)
    
    variable_summaries(wd, "weights")
    variable_summaries(bd, "biases")
    reg=tf.multiply(beta,tf.nn.l2_loss(wd))
    return dense_out,reg
            
def mats_4_err_calc(locate):
    w0 = matlib.repmat(np.identity(3),1,locate)
    print 'w0', w0.shape
    w1 = np.zeros((3*locate,locate))
    for l in range(0,locate):
        w1[3*l:3*(l+1),l]=1./3.
    print 'w1', w1.shape
    print w1
    return w0,w1

def variable_summaries(var, name):
  with tf.variable_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.variable_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.summary.scalar('sttdev/' + name, stddev)
      tf.summary.scalar('max/' + name, tf.reduce_max(var))
      tf.summary.scalar('min/' + name, tf.reduce_min(var))
      tf.summary.histogram('histogram/'+name, var)

def sph2cart(r,th,ph):
    with tf.variable_scope("convert"):
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
        if locate is not False:
            print "For nearest-neighbor technique, n_out must be 3"
            self.n_out=3
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
        if self.cnn is not False:
            self.n_dense=int((self.meas_dims[0]-self.k_conv+1)/self.k_pool-self.k_conv+1)*int((self.meas_dims[1]-self.k_conv+1)/self.k_pool-self.k_conv+1)*self.n_conv2
        else:
            self.n_dense=self.meas_dims
        print "n_dense: ", self.n_dense
        self.std_dense = 1.#100./self.n_dense
        
        self.w0, self.w1 = mats_4_err_calc(locate)
        
    def cnn_nn(self):        
        with tf.variable_scope("cnn_nn"):
            self.measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
            # Store layers weight & bias
            with tf.variable_scope('cnn'):
                cnn_out, reg1 = cnn_2layers(self.measfold, self.k_conv, self.n_chan_in, self.n_conv1, self.n_conv2,self.dropoutPH, self.betaPH)
                dense_cnn = tf.reshape(cnn_out, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
            with tf.variable_scope('dense'):
                dense_out, reg2 = dense(dense_cnn, self.n_dense, self.n_out,self.dropoutPH, self.betaPH,self.std_dense)
            self.logits = dense_out
            dense_last = tf.reshape(dense_out, [-1, self.n_steps, self.n_out])
            self.logits_last = dense_last[:,-1,:]
            self.reg=reg1+reg2

    def rnn_nn(self):
        with tf.variable_scope("rnn_nn"):
            dense_in = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            with tf.variable_scope('dense_in'):
                dense_out,reg1 = dense(dense_in, self.n_dense, self.n_lstm, self.dropoutPH, self.betaPH, 1.0)

            with tf.variable_scope('rnn'):
                last, outs = rnn(dense_out,self.n_lstm,self.n_steps,self.n_layer,self.dropoutPH, self.betaPH)

            with tf.variable_scope('dense_all'):
                self.logits, reg2 = dense(outs, self.n_lstm, self.n_out, self.dropoutPH, self.betaPH, self.std_dense)             

            with tf.variable_scope('dense_last'):
                self.logits_last, reg3= dense(last, self.n_lstm, self.n_out, self.dropoutPH, self.betaPH, self.std_dense)

            self.reg=reg1+reg3
        
    def cnn_rnn(self):
        with tf.variable_scope("cnn_rnn"):
            self.measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
            # Store layers weight & bias
            with tf.variable_scope('cnn'):
                cnn_out,reg1 = cnn_2layers(self.measfold, self.k_conv, self.n_chan_in, self.n_conv1, self.n_conv2, self.dropoutPH, self.betaPH)
                dense_cnn = tf.reshape(cnn_out, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input

            with tf.variable_scope('dense'):
                dense_out,reg2 = dense(dense_cnn, self.n_dense, self.n_lstm,self.dropoutPH, self.betaPH, 1.0)

            with tf.variable_scope('rnn'):
                last, outs = rnn(dense_out,self.n_lstm,self.n_steps,self.n_layer,self.dropoutPH, self.betaPH)

            with tf.variable_scope('dense_all'):
                self.logits,reg3 = dense(outs, self.n_lstm, self.n_out, self.dropoutPH, self.betaPH, self.std_dense)             

            with tf.variable_scope('dense_last'):
                self.logits_last,reg4= dense(last, self.n_lstm, self.n_out, self.dropoutPH, self.betaPH, self.std_dense)

            self.reg=reg1+reg2+reg4

    def mlp(self):
        with tf.variable_scope("mlp"):
            dense_in = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            with tf.variable_scope('dense_in'):
                dense_out,reg1 = dense(dense_in, self.n_dense, self.n_lstm, self.dropoutPH, self.betaPH, 1.0)
            with tf.variable_scope('dense_out'):
                self.logits,reg2 = dense(dense_out, self.n_lstm, self.n_out, self.dropoutPH, self.betaPH, self.std_dense)
            dense_last = tf.reshape(dense_out, [-1, self.n_steps, self.n_out])
            self.logits_last = dense_last[:,-1,:]

            self.reg=reg1+reg2

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
        with tf.variable_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.saver = tf.train.Saver()
            
    def initializer(self):
        self.init_step = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()
    def cost(self): 
        self.W0 = tf.constant(self.w0, dtype=tf.float32)
        self.W1 = tf.constant(self.w1, dtype=tf.float32)
        
        self.qtrainPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.n_obs), name="qtrain")

        with tf.variable_scope("cost_parse"):
            self.qtrain_unflat = tf.reshape(self.qtrainPH,[-1,self.n_obs])#b*nxp
            self.qtrain_last = self.qtrainPH[:,-1,:]#bxp
            if self.locate is not False:
                self.qhat = self.logits
                self.qhat_last = self.logits_last
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")

            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)
        
        with tf.variable_scope('cost'):
            if ((self.rnn is False) and (self.cnn is not 'fft')):                
                B=tf.argmax(self.qtrain_unflat,1)
                qtrain_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
        
                self.cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, qtrain_OH),name="cross"),self.reg)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.A,B),tf.float32),name="accuracy")

                
                if self.locate is not False:
                    with tf.variable_scope('rep_outputs'):
                        qhat_rep = tf.matmul(self.qhat,self.W0)#b*nx3x3x3l
                        SE = tf.matmul(tf.square(tf.sub(qhat_rep,self.qtrain_unflat)),self.W1)#b*nx3lx3lxl
                        SEbn = tf.reduce_min(SE,1)#b*n
                        SSE = tf.reduce_mean(SEbn)#1
                        self.rmse = tf.add(tf.sqrt(SSE,name="rmse"),self.reg)
                else:
                    self.rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,self.qtrain_unflat))),name="rmse"),self.reg)

                if self.locate is not False:
                    self.cost = self.rmse 
                else:
                    self.cost = self.cross

            else:#yes rnn or cnn is fft
                B=tf.argmax(self.qtrain_unflat,1)#b*nx1
                qtrain_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
        
                BB=tf.argmax(self.qtrain_last,1)#bx1
                self.qtrain_last_OH = tf.one_hot(BB,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
                
                self.cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits_last, self.qtrain_last_OH),name="cross_last"),self.reg)
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(self.AA,BB),tf.float32),name="accuracy_last")
                
                if self.locate is not False:
                    with tf.variable_scope('rep_outputs'):
                        qhat_rep = tf.matmul(self.qhat_last,self.W0)#bx3x3x3l
                        SE = tf.matmul(tf.square(tf.sub(qhat_rep,self.qtrain_last)),self.W1)#bx3lx3lxl
                        SEnb = tf.reduce_min(SE,axis=1)#b
                        SSE = tf.reduce_mean(SEnb)#1
                        self.rmse_last = tf.add(tf.sqrt(SSE,name="rmse"),self.reg)
                else:
                    self.rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qtrain_last,self.qhat_last))),name="rmse_last"),self.reg)
                
                if self.locate is not False:
                    self.cost = self.rmse_last 
                else:
                    self.cost = self.cross_last

        with tf.variable_scope('summaries'):
            self.train_summary = tf.summary.scalar('mean/train_cost', self.cost)
            self.valid_summary = tf.summary.scalar('mean/valid_cost', self.cost)
