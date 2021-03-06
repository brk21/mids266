import tensorflow as tf
import numpy as np

def wordids_to_tensors(wordids, embedding_dim, vocab_size, seed=0):
    '''Convert a list of wordids into embeddings and biases.

    This function creates a variable for the embedding matrix, dimension |E x V|
    and a variable to hold the biases, dimension |V|.

    It returns an op that will accept the output of "wordids" op and lookup
    the corresponding embedding vector and bias in the table.

    Args:
      - wordids |W|: a tensor of wordids
      - embedding_dim, E: a scalar value of the # of dimensions in which to embed words
      - vocab_size |V|: # of terms in the vocabulary

    Returns:
      - a tuple (w, b, m) where w is a tensor of word embeddings and b is a vector
        of biases.  w is |W x E| and b is |W|.  m is the full |V x E| embedding matrix.
        Each of these should contain values of type tf.float32.
    '''
    # START YOUR CODE
    
    m = tf.get_variable(name="m", shape=[vocab_size,embedding_dim],
                        dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-1, maxval=1,seed=seed))
    w = tf.nn.embedding_lookup(m, wordids)
    b = tf.get_variable(name="b", shape=[vocab_size],
                       dtype=tf.float32,initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    b1 = tf.nn.embedding_lookup(b, wordids)
    
    return (w,b1,m)
    
    # END YOUR CODE


def example_weight(Xij, x_max, alpha):
    '''Scale the count according to Equation (9) in the Glove paper.

    This runs as part of the TensorFlow graph.  You must do this with
    TensorFlow ops.

    Args:
      - Xij: a |batch| tensor of counts.
      - x_max: a scalar, see paper.
      - alpha: a scalar, see paper.

    Returns:
      - A vector of corresponding weights.
    '''
    # START YOUR CODE
    y = tf.constant(1.0)
    return tf.minimum(tf.pow(tf.div(Xij,x_max),alpha),y,name="condition")
    # END YOUR CODE


def loss(w, b, w_c, b_c, c):
    '''Compute the loss for each of training examples.

    Args:
      - w |batch_size x embedding_dim|: word vectors for the batch
      - b |batch_size|: biases for these words
      - w_c |batch_size x embedding_dim|: context word vectors for the batch
      - b_c |batch_size|: biases for context words
      - c |batch_size|: # of times context word appeared in context of word

    Returns:
      - loss |batch_size|: the loss of each example in the batch
    '''
    # START YOUR CODE
    mul = tf.mul(w,w_c)
    dotp = tf.reduce_sum(mul,1)
    loss = example_weight(c,100,.75)*tf.square(tf.sub(tf.add(tf.add(dotp, b),b_c),tf.log(c)))
    return loss
    # END YOUR CODE
