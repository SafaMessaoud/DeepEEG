
def model4(self):
  self.trial_data = tf.reshape(self.trial_data, [self.config.batch_size,self.config.nb_channels,self.config.nb_freq,self.config.nb_time_windows])
  trial_data = tf.transpose(self.trial_data, [3,0,1,2])
  
  with tf.variable_scope("model4") as model4:
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=model4_lstm_state_size, state_is_tuple=False)
    
    state1 = tf.zeros([self.config.batch_size, lstm_cell.state_size])
    
    for time  in range(self.config.nb_time_windows) :
        trial_data_t =tf.slice(trial_data, [time,0,0,0], [1,-1,-1,-1])
        trial_data_t =tf.reshape(trial_data_t, [self.config.batch_size, self.config.nb_channels,self.config.nb_freq ])
        elec_current_time=tf.reduce_max(trial_data_t, axis=1)
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



 
