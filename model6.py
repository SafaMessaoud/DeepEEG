def baseline_elec_attention(self,elec1,elec2,state_size):
    
    self.config.model6_att_elec_dim=state_size+self.config.nb_freq

    att_embed_W1 = tf.Variable(tf.random_uniform([self.config.model6_att_elec_dim, self.config.model6_att_elec_dim], -0.1,0.1), name='model6_att_W_elec1')
    att_embed_W2 = tf.Variable(tf.random_uniform([self.config.model6_att_elec_dim, 1], -0.1,0.1), name='model6_att_W_elec2')
    att_embed_b1 = tf.Variable(tf.zeros([self.config.model6_att_elec_dim]), name='model6_att_b_elec1')
    att_embed_b2 = tf.Variable(tf.zeros([1]), name='model6_att_b_elec2')

    feat_tensor = tf.reshape(elec_current_time_x, [-1,self.config.model6_att_elec_dim])

    e = tf.nn.relu(tf.matmul(feat_tensor, att_embed_W1)+att_embed_b1 )  # [batch * nb_electrodes, embed_att_elec_dim]
    e = tf.matmul(e,self.att_embed_W2)+ att_embed_b2  # [batch * nb_electrodes, 1]
    e = tf.reshape(e, [ self.config.batch_size, max_nb_channels ])  # [batch , nb_electrodes]
       
    alphas = tf.nn.softmax(e)    
    alphas = tf.reshape(alphas, [ self.config.batch_size, max_nb_channels,1 ])

    attention_list = tf.multiply(alphas,elec_current_time)
    output = tf.reduce_sum(attention_list,1)  #[batch_size,electrode_representation_size]

    return output





def model6(self):
  self.trial_data = tf.reshape(self.trial_data, [self.config.batch_size,self.config.nb_channels,self.config.nb_freq,self.config.nb_time_windows])
  trial_data = tf.transpose(self.trial_data, [3,0,1,2])
  
  with tf.variable_scope("model6") as model6:
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=model6_lstm_state_size, state_is_tuple=False)
    
    state1 = tf.zeros([self.config.batch_size, lstm_cell.state_size])
    
    for time  in range(self.config.nb_time_windows) :
        trial_data_t =tf.slice(trial_data, [time,0,0,0], [1,-1,-1,-1])
        trial_data_t =tf.reshape(trial_data_t, [self.config.batch_size, self.config.nb_channels,self.config.nb_freq ])
        
        hidden_state_tensor = tf.tile(state1, [1,self.config.nb_channels], name=None)
        hidden_state_tensor=tf.reshape( hidden_state_tensor,[self.config.batch_size,self.config.nb_channels ,lstm_cell.state_size ] )
        
        trial_data_t2=tf.concat([trial_data_t, hidden_state_tensor], 2)
        
        elec_current_time=model6_elec_attention(self,trial_data_t,trial_data_t2, lstm_cell.state_size)
        elec_current_time=tf.reshape(elec_current_time, [ self.config.batch_size , -1])

          
        with tf.variable_scope('lstm1',reuse=True if time > 0 else None) as scope1:
          output1, state1 = lstm_cell( inputs=elec_current_time, state=state1 )
          
  with tf.variable_scope("logits") as logits_scope:
    logits = tf.contrib.layers.fully_connected(
    inputs=output1,
    num_outputs=self.config.num_classes,
    activation_fn=None,
    weights_initializer=self.weight_initializer,
    scope='model6_log')
  
  return logits
