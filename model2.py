def rnn_elec_architecture(self,data):

  #current data shape:[batch_size*nb_channels,nb_freq ,nb_time_windows ])
  #reshape to: [nb_time_windows,batch_size*nb_channels,nb_freq]
  rnn_inputs = tf.transpose(data, [2,0,1])
  
  cell = tf.contrib.rnn.LSTMCell(num_units=self.config.model2_rnn_state_size, state_is_tuple=False)
  state1 = tf.zeros([self.config.batch_size*self.config.nb_channels , cell.state_size])
  
  for time  in range(self.config.nb_time_windows) :
    rnn_input_current_time=tf.slice(rnn_inputs, [time,0,0], [1,-1,-1])
    rnn_input_current_time=tf.reshape(rnn_input_current_time,[ self.config.batch_size*self.config.nb_channels , self.config.nb_freq  ])
    
    with tf.variable_scope('lstm1',reuse=True if time > 0 else None) as scope1:
      output1, state1 = cell( inputs=rnn_input_current_time, state=state1 )
      
  return output1

def cnn_elec_architecture(self,data):
  x=tf.expand_dims(data, 3) 
  
  kernel_width=model2_kernel_width
  kernel_hight=model2_kernel_hight
  stride_width=model2_stride_width
  
  output1=tf.contrib.layers.convolution2d(
  inputs= x,
  num_outputs= model2_output1_dim , 
  kernel_size=[kernel_width,kernel_hight],
  stride=stride_width,
  normalizer_fn=self.normalizer_fn,
  padding='SAME',
  activation_fn=tf.nn.relu,
  weights_initializer=self.weight_initializer,
  weights_regularizer = self.weights_regularizer,
  biases_initializer=y.initialized_value(),
  trainable=True,
  scope='model2_conv1')
  
  max_pool=tf.nn.max_pool(output1, 
  ksize=[1, 2, 2, 1],
  strides=[1, 2, 2, 1],
  padding='SAME',
  name='model2_pool')
  
  reshape = tf.reshape(max_pool, [batch_size, -1])
  
  fc = tf.contrib.layers.fully_connected(
  inputs=reshape,
  num_outputs=self.config.model2_cnn_representation_size,
  activation_fn=tf.nn.relu,
  weights_initializer=self.weight_initializer,
  weights_regularizer = self.weights_regularizer,
  biases_initializer=None,
  scope='model2_fc')
  
  return fc


def model2(self):
  self.trial_data = tf.reshape(self.trial_data, [self.config.batch_size,self.config.nb_channels,self.config.nb_freq,self.config.nb_time_windows])
  
  with tf.variable_scope("model2") as model2_scope:
    trial_data_t=tf.reshape(self.trial_data, [self.config.batch_size*self.config.nb_channels,self.config.nb_freq ,self.config.nb_time_windows ])
    
    #RNN
    rnn_output=rnn_elec_architecture(self,trial_data_t) 
    
    #reshape
    rnn_out=tf.reshape(rnn_output, [ batch_size, max_nb_channels , -1])
    
    #1D Conv    
    elec_current_time=cnn_elec_architecture(self,rnn_out) 
    elec_current_time=tf.reshape(elec_current_time, [ batch_size , -1])
    
    
  with tf.variable_scope("model2_logits") as logits_scope:
    logits = tf.contrib.layers.fully_connected(
    inputs=elec_current_time,
    num_outputs=self.config.num_classes,
    activation_fn=None,
    weights_initializer=self.weight_initializer,
    scope='model2_log')
  
  return logits
