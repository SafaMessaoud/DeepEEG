def rnn_elec_architecture(self,data):

  #current data shape:[batch_size*nb_channels,nb_freq ,nb_time_windows ])
  #reshape to: [nb_time_windows,batch_size*nb_channels,nb_freq]
  rnn_inputs = tf.transpose(data, [2,0,1])
  
  cell = tf.contrib.rnn.LSTMCell(num_units=self.config.model0_rnn_state_size, state_is_tuple=False)
  state1 = tf.zeros([self.config.batch_size*self.config.nb_channels , cell.state_size])
  
  for time  in range(self.config.nb_time_windows) :
    rnn_input_current_time=tf.slice(rnn_inputs, [time,0,0], [1,-1,-1])
    rnn_input_current_time=tf.reshape(rnn_input_current_time,[ self.config.batch_size*self.config.nb_channels , self.config.nb_freq  ])
    
    with tf.variable_scope('lstm1',reuse=True if time > 0 else None) as scope1:
      output1, state1 = cell( inputs=rnn_input_current_time, state=state1 )
      
  return output1





def model0(self):
  self.trial_data = tf.reshape(self.trial_data, [self.config.batch_size,self.config.nb_channels,self.config.nb_freq,self.config.nb_time_windows])
  
  with tf.variable_scope("model0") as model0_scope:
    trial_data_t=tf.reshape(self.trial_data, [self.config.batch_size*self.config.nb_channels,self.config.nb_freq ,self.config.nb_time_windows ])
    
    #RNN
    rnn_output=rnn_elec_architecture(self,trial_data_t) 
    
    #reshape
    rnn_out=tf.reshape(rnn_output, [ batch_size, max_nb_channels , -1])
    
    #maxpooling
    elec_current_time=tf.reduce_max(rnn_out, axis=1)
    elec_current_time=tf.reshape(elec_current_time, [ batch_size , -1])
    
    
  with tf.variable_scope("logits") as logits_scope:
    logits = tf.contrib.layers.fully_connected(
    inputs=elec_current_time,
    num_outputs=self.config.num_classes,
    activation_fn=None,
    weights_initializer=self.weight_initializer,
    scope='log')
  
  return logits
    
   
