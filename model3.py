def rnn_elec_architecture(self,data):

  #current data shape:[batch_size*nb_channels,nb_freq ,nb_time_windows ])
  #reshape to: [nb_time_windows,batch_size*nb_channels,nb_freq]
  rnn_inputs = tf.transpose(data, [2,0,1])
  
  cell = tf.contrib.rnn.LSTMCell(num_units=self.config.model3_rnn_state_size, state_is_tuple=False)
  state1 = tf.zeros([self.config.batch_size*self.config.nb_channels , cell.state_size])
  
  for time  in range(self.config.nb_time_windows) :
    rnn_input_current_time=tf.slice(rnn_inputs, [time,0,0], [1,-1,-1])
    rnn_input_current_time=tf.reshape(rnn_input_current_time,[ self.config.batch_size*self.config.nb_channels , self.config.nb_freq  ])
    
    with tf.variable_scope('lstm1',reuse=True if time > 0 else None) as scope1:
      output1, state1 = cell( inputs=rnn_input_current_time, state=state1 )
      
  return output1

def model3_elec_attention(self,elec_current_time):
  att_embed_W1 = tf.Variable(tf.random_uniform([self.model3_att_elec_dim, self.model3_att_elec_dim], -0.1,0.1), name='model3_att_W_elec1')
  att_embed_W2 = tf.Variable(tf.random_uniform([self.model3_att_elec_dim, 1], -0.1,0.1), name='model3_att_W_elec2')
  att_embed_b1 = tf.Variable(tf.zeros([self.model3_att_elec_dim]), name='model3_att_b_elec1')
  att_embed_b2 = tf.Variable(tf.zeros([1]), name='model3_att_b_elec2')
  
  feat_tensor = tf.reshape(elec_current_time, [-1,self.model3_att_elec_dim])
  
  e = tf.nn.relu(tf.matmul(feat_tensor, att_embed_W1)+att_embed_b1 )  # [batch * nb_electrodes, embed_att_elec_dim]
  e = tf.matmul(e,att_embed_W2)+ att_embed_b2  # [batch * nb_electrodes, 1]
  e = tf.reshape(e, [ batch_size, max_nb_channels ])  # [batch , nb_electrodes]
  e_hat_exp =  tf.exp(e) # [batch , nb_electrodes]
  
  #compute the dominator
  denomin = tf.reduce_sum(e_hat_exp,1) # [batch]
  denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
  denomin = tf.expand_dims(denomin, 1) 
  denomin = tf.tile(denomin, [1,self.config.nb_channels])  #expand denomin: [batch,nb_electrodes]
  
  #generate the attention weights
  alphas = tf.div(e_hat_exp,denomin)
  alphas = tf.reshape(alphas, [ self.config.batch_size, self.config.nb_channels,1 ])
  
  #attention_list : multiply alphas [batch,nb_electrodes,1] with [batch,nb_electrodes,electrode_representation_size]
  attention_list = tf.multiply(alphas,elec_current_time)
  output = tf.reduce_sum(attention_list,1)  #[batch_size,electrode_representation_size]
  
  return output




def model3(self):
  self.trial_data = tf.reshape(self.trial_data, [self.config.batch_size,self.config.nb_channels,self.config.nb_freq,self.config.nb_time_windows])
  
  with tf.variable_scope("model3") as model3_scope:
    trial_data_t=tf.reshape(self.trial_data, [self.config.batch_size*self.config.nb_channels,self.config.nb_freq ,self.config.nb_time_windows ])
    
    #RNN
    rnn_output=rnn_elec_architecture(self,trial_data_t) 
    
    #reshape
    rnn_out=tf.reshape(rnn_output, [ batch_size, max_nb_channels , -1])
    
    #Attention
    elec_current_time=model3_elec_attention(self,rnn_out)
    elec_current_time=tf.reshape(elec_current_time, [ batch_size , -1])
    
    
  with tf.variable_scope("model3_logits") as logits_scope:
    logits = tf.contrib.layers.fully_connected(
    inputs=elec_current_time,
    num_outputs=self.config.num_classes,
    activation_fn=None,
    weights_initializer=self.weight_initializer,
    scope='model3_log')
  
  return logits
