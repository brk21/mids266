import time

import tensorflow as tf
import numpy as np

def matmul3d(X, W):
  """Wrapper for tf.matmul to handle a 3D input tensor X.
  Will perform multiplication along the last dimension.

  Args:
    X: [m,n,k]
    W: [k,l]

  Returns:
    XW: [m,n,l]
  """
  Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
  XWr = tf.matmul(Xr, W)
  newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
  return tf.reshape(XWr, newshape)

def MakeFancyRNNCell(H, keep_prob, num_layers=1):
  """Make a fancy RNN cell.

  Use tf.nn.rnn_cell functions to construct an LSTM cell.
  Initialize forget_bias=0.0 for better training.

  Args:
    H: hidden state size
    keep_prob: dropout keep prob (same for input and output)
    num_layers: number of cell layers

  Returns:
    (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
  """
  #### YOUR CODE HERE ####
  cell = None  # replace with something better

  # Solution
  cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
  cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                       output_keep_prob=keep_prob)
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

  #### END(YOUR CODE) ####
  return cell

class RNNLM(object):

  def __init__(self, V, H, num_layers=1):
    """Init function.

    This function just stores hyperparameters. You'll do all the real graph
    construction in the Build*Graph() functions below.

    Args:
      V: vocabulary size
      H: hidden state dimension
      num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
    """
    # Model structure; these need to be fixed for a given model.
    self.V = V
    self.H = H
    self.num_layers = 1

    # Training hyperparameters; these can be changed with feed_dict,
    # and you may want to do so during training.
    with tf.name_scope("Training_Parameters"):
      self.learning_rate_ = tf.constant(0.1, name="learning_rate")
      self.dropout_keep_prob_ = tf.constant(0.5, name="dropout_keep_prob")
      # For gradient clipping, if you use it.
      # Due to a bug in TensorFlow, this needs to be an ordinary python
      # constant instead of a tf.constant.
      self.max_grad_norm_ = 5.0
        
  def BuildCoreGraph(self):
    """Construct the core RNNLM graph, needed for any use of the model.

    This should include:
    - Placeholders for input tensors (input_w, initial_h, target_y)
    - Variables for model parameters
    - Tensors representing various intermediate states
    - A Tensor for the output logits (logits_)
    - A scalar loss function (loss_)

    Your loss function should return a *scalar* value that represents the
    _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
    tf.reduce_mean).

    You shouldn't include training or sampling functions here; you'll do this
    in BuildTrainGraph and BuildSampleGraph below.

    We give you some starter definitions for input_w_ and target_y_, as well 
    as a few other tensors that might help. We've also added dummy values for 
    initial_h_, logits_, and loss_ - you should re-define these in your code as 
    the appropriate tensors. See the in-line comments for more detail.
    """
    # Input ids, with dynamic shape depending on input.
    # Should be shape [batch_size, max_time] and contain integer word indices.
    self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

    # Initial hidden state. You'll need to overwrite this with cell.zero_state
    # once you construct your RNN cell.
    self.initial_h_ = None
    
    ### WHAT IS A HIDDEN STATE? WHY IS IT ONLY A SCALAR NUMBER? WHY NOT A PALCEHOLDER?

    # Output logits, which can be used by loss functions or for prediction.
    # Overwrite this with an actual Tensor of shape [batch_size, max_time]
    self.logits_ = None

    # Should be the same shape as inputs_w_
    self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

    # Replace this with an actual loss function
    self.loss_ = None

    # Get dynamic shape info from inputs
    with tf.name_scope("batch_size"):
      self.batch_size_ = tf.shape(self.input_w_)[0]
    with tf.name_scope("max_time"):
      self.max_time_ = tf.shape(self.input_w_)[1]

    # Get sequence length from input_w_.
    # This will be a vector with elements ns[i] = len(input_w_[i])
    # You can override this in feed_dict if you want to have different-length
    # sequences in the same batch, although you shouldn't need to for this
    # assignment.
    self.ns_ = tf.tile([self.max_time_], [self.batch_size_,], name="ns")

    #### YOUR CODE HERE ####

    # Construct embedding layer
    with tf.name_scope("embedding_layer"):
        C_ = tf.get_variable(name="C", shape=[self.V,self.H],
                            dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    # embedding_lookup gives shape (batch_size, N, M)
        x_ = tf.reshape(tf.nn.embedding_lookup(C_, self.input_w_), 
                    [-1, self.V,self.H], name="x")
        b = tf.get_variable(name="b", shape=[self.V],
                           dtype=tf.float32,initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        b1 = tf.nn.embedding_lookup(b, self.input_w_)
        


    # Construct RNN/LSTM cell and recurrent layer
    with tf.name_scope("LSTM_RNN_layer"):
        cell = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
        self.initial_h_ = cell.zero_state
        self.output = tf.nn.dynamic_rnn(cell, self.initial_h_, dtype=tf.float32)
        ### IS THIS THE FUNCTION THAT I SHOULD BE USING? WHAT'S THE DIFFERENCE BETWEEN THE INIITAL_H AND THE STATE OF THE CELL? SHOULD WE BE DEFINING 

    # Softmax output layer, over vocabulary
    # Hint: use the matmul3d() helper here.
    ### IS THE OUTPUT OF THE DYNAMIC RNN ABOVE THE SAME AS W23 BELOW? THAT'S HOW I HAVE IT RIGHT NOW
    with tf.name_scope("output_layer"):
        self.b3_ = tf.Variable(tf.zeros([self.V,], dtype=tf.float32), name="b3")
        # Concat [h x] and [W2 W3]
        self.hx_ = tf.concat(1, [self.initial_h_, x_], name="hx")
        #W23_ = tf.concat(0, [W2_, W3_], name="W23")
        self.logits_ = tf.add(tf.matmul3d(self.hx_, self.output), b1, name="logits")
        ### THE SOFTMAX HAPPENS IN THE LOSS COMPUTATION BELOW, RIGHT? THIS IS JUST THE MATRIX MULTIPLICATION. WHERE IS THE CONCATENATION? DOES ALL OF THAT HAPPEN IN THE APIS?

    # Loss computation (true loss, for prediction)
    with tf.name_scope("Loss_Computation"):
        self.loss_ = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_, self.target_y_)
                                ,name='loss')
        ### WHAT IS THE DIFFERENCE BETWEEN THIS AND THE LOSS ABOVE?

    #### END(YOUR CODE) ####


  def BuildTrainGraph(self):
    """Construct the training ops.

    You should define:
    - train_loss_ (optional): an approximate loss function for training
    - train_step_ : a training op that can be called once per batch

    Your loss function should return a *scalar* value that represents the
    _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
    tf.reduce_mean).
    """
    # Replace this with an actual training op
    self.train_step_ = tf.no_op(name="dummy")

    # Replace this with an actual loss function
    self.train_loss_ = None

    #### YOUR CODE HERE ####
    # Define loss function(s)
    with tf.name_scope("Train_Loss"):
      # Placeholder: replace with a sampled loss
      self.train_loss = tf.reduce_sum(tf.nn.sampled_softmax_loss(tf.transpose(self.output),
                                                                 self.b3_, self.hx_, 
                                             labels=tf.expand_dims(self.target_y_, 1), 
                                             num_sampled=100, num_classes=self.V,
                                             name="per_example_sampled_softmax_loss"),
                                    name="sampled_softmax_loss")


    # Define optimizer and training op
    with tf.name_scope("Training"):
      self.optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
      self.train_step_ = optimizer_.minimize(self.train_loss_)  
    #### END(YOUR CODE) ####


  def BuildSamplerGraph(self):
    """Construct the sampling ops.

    You should define pred_samples_ to be a Tensor of integer indices for
    sampled predictions for each batch element, at each timestep.

    Hint: use tf.multinomial, along with a couple of calls to tf.reshape
    """
    # Replace with a Tensor of shape [batch_size, max_time, 1]
    self.pred_samples_ = None

    #### YOUR CODE HERE ####
    with tf.name_scope("Prediction"):
        self.pred_samples_ = tf.multinomial(self.logits_,self.batch_size_,name ="pred_random")
        self.pred_proba_ = tf.nn.softmax(self.logits_, name="pred_proba")
        self.pred_max_ = tf.argmax(self.logits_, 1, name="pred_max")
    #### END(YOUR CODE) ####

