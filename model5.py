
def cnn_elec_architecture(self,data):
  x=tf.expand_dims(data, 1) #[batch_size,1,nb_elec,elec_representation]
  x=tf.transpose(x, perm=[0,1,3,2]) #[batch_size,1,nb_freq,nb_elec]
       
  kernel_hight=1
  kernel_width=1
  stride_width=1
            
  output1=tf.contrib.layers.convolution2d(
  inputs= x,
  num_outputs= model5_output1_dim , 
  kernel_size=[kernel_width,kernel_hight],
  stride=stride_width,
  normalizer_fn=self.normalizer_fn,
  padding='SAME',
  activation_fn=tf.nn.relu,
  weights_initializer=self.weight_initializer,
  weights_regularizer = self.weights_regularizer,
  biases_initializer=y.initialized_value(),
  trainable=True,
  scope='model5_conv1')
       
  max_pool=tf.nn.max_pool(output1,
  ksize=[1, 1, 1, 1],
  strides=[1, 1, 1, 1],
  padding='SAME',
  name='model5_pool1')
    
    
  output2=tf.contrib.layers.convolution2d(
  inputs= max_pool,
  num_outputs= model5_output2_dim ,  
  kernel_size=[kernel_width,kernel_hight],
  stride=1,
  normalizer_fn=self.normalizer_fn,
  padding='SAME',
  activation_fn=tf.nn.relu,
  weights_initializer=self.weight_initializer,
  weights_regularizer = self.weights_regularizer,
  biases_initializer=y.initialized_value(),
  trainable=True,
  scope='model5_conv2')
    
  reshape = tf.reshape(output2, [self.config.batch_size, -1])
    
  fc = tf.contrib.layers.fully_connected(
  inputs=reshape,
  num_outputs=model5_cnn_representation_size,
  activation_fn=tf.nn.relu,
  weights_initializer=self.weight_initializer,
  weights_regularizer = self.weights_regularizer,
  biases_initializer=None,
  scope='model5_fc')
  
  return fc



def model5(self):
  self.trial_data = tf.reshape(self.trial_data, [self.config.batch_size,self.config.nb_channels,self.config.nb_freq,self.config.nb_time_windows])
  trial_data = tf.transpose(self.trial_data, [3,0,1,2])
  
  with tf.variable_scope("model5") as model5:
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=model5_lstm_state_size, state_is_tuple=False)
    
    state1 = tf.zeros([self.config.batch_size, lstm_cell.state_size])
    
    for time  in range(self.config.nb_time_windows) :
        trial_data_t =tf.slice(trial_data, [time,0,0,0], [1,-1,-1,-1])
        trial_data_t =tf.reshape(trial_data_t, [self.config.batch_size, self.config.nb_channels,self.config.nb_freq ])
        
        elec_current_time=cnn_elec_architecture(self,trial_data_t)
        elec_current_time=tf.reshape(elec_current_time, [ self.config.batch_size , -1])
                
        with tf.variable_scope('lstm1',reuse=True if time > 0 else None) as scope1:
          output1, state1 = lstm_cell( inputs=elec_current_time, state=state1 )
          
  with tf.variable_scope("logits") as logits_scope:
    logits = tf.contrib.layers.fully_connected(
    inputs=output1,
    num_outputs=self.config.num_classes,
    activation_fn=None,
    weights_initializer=self.weight_initializer,
    scope='log')
  
  return logits
