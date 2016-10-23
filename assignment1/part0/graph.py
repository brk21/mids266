import numpy as np
import tensorflow as tf
tf.reset_default_graph()

class AddTwo(object):
    def __init__(self):
        # If you are constructing more than one graph within a Python kernel
        # you can either tf.reset_default_graph() each time, or you can
        # instantiate a tf.Graph() object and construct the graph within it.
        # Hint: Recall from live sessions that TensorFlow
        # splits its models into two chunks of code:
        # - construct and keep around a graph of ops
        # - execute ops in the graph
        #
        # Construct your graph in __init__ and run the ops in Add.
        #
        # We make the separation explicit in this first subpart to
        # drive the point home.  Usually you will just do them all
        # in one place, including throughout the rest of this assignment.
        #
        # Hint:  You'll want to look at tf.placeholder and sess.run.

        # START YOUR CODE
        self.init_ = tf.initialize_all_variables()
        self.session = tf.Session(config=tf.ConfigProto(device_filters="/cpu:0"))
        # END YOUR CODE

    def Add(self, x, y):
        # START YOUR CODE
        if isinstance(x,np.ndarray):
            X_ = tf.placeholder(tf.int64,shape=x.shape, name="X")
        elif isinstance(x,int):
            X_ = tf.placeholder(tf.int64, name="X")
        if isinstance(y,np.ndarray):
            Y_ = tf.placeholder(tf.int64,shape=y.shape, name ="Y")
        elif isinstance(y,int):
            Y_ = tf.placeholder(tf.int64, name ="Y")
        else: 
            return "input is not an integer or array"
        summation = X_ + Y_
        self.session.run(self.init_)
        final_sum = self.session.run([summation], 
                            feed_dict={X_: x, Y_: y})
        if len(final_sum) == 1:
            return final_sum[0]
        else:
            return final_sum
        # END YOUR CODE

def affine_layer(hidden_dim, x, seed=0):
    # x: a [batch_size x # features] shaped tensor.
    # hidden_dim: a scalar representing the # of nodes. Output dimension of the Affine Layer
    # seed: use this seed for xavier initialization.
    # START YOUR CODE
    seeded = seed
    with tf.name_scope('affine_parameters'):
        W_ = tf.get_variable(name="W", shape=[x.get_shape()[-1],hidden_dim],
                initializer=tf.contrib.layers.xavier_initializer(seed=seeded), trainable=True)
        b_ = tf.get_variable(name="b", shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0), trainable=True)
        # Create variable named "biases".
    
    with tf.name_scope('outputs'):
        matrix_mult = tf.add(tf.matmul(x, W_), b_)
    return matrix_mult
    # END YOUR CODE

def fully_connected_layers(hidden_dims, x):
    # hidden_dims: A list of the width of the hidden layer.
    # x: the initial input with arbitrary dimension.
    # To get the tests to pass, you must use relu(.) as your element-wise nonlinearity.
    #### Iterate over all of the hidden dimensions, using the Affine Layer
    #### Make sure that your variable scope name is different for each layer
    # Hint: see tf.variable_scope - you'll want to use this to make each layer 
    # unique.

    # START YOUR CODE
    output = []
    for i,dims in enumerate(hidden_dims):
        with tf.name_scope('connected_layer_%s'% i):
            with tf.variable_scope("mylayer_%s" % i) as scope:
                if len(hidden_dims) == 0:
                    y_out = affine_layer(dims,x)
                    output = y_out
                elif i == 0 and len(hidden_dims) > 0:
                    y_out = tf.nn.relu(affine_layer(dims,x))
                    output = y_out
                elif i == len(hidden_dims):
                    y_out = affine_layer(dims,y_out)
                    output = y_out
                else:
                    y_out = tf.nn.relu(affine_layer(dims,y_out))
                    output = y_out
    return output
    # END YOUR CODE

def train_nn(X, y, X_test, hidden_dims, batch_size, num_epochs, learning_rate):
    # Train a neural network consisting of fully_connected_layers
    # to predict y.  Use sigmoid_cross_entropy_with_logits loss between the
    # prediction and the label.
    # Return the predictions for X_test.
    # X: train features
    # Y: train labels
    # X_test: test features
    # hidden_dims: same as in fully_connected_layers
    # learning_rate: the learning rate for your GradientDescentOptimizer.

    # Construct the placeholders.
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=[None, X.shape[-1]])
    y_ph = tf.placeholder(tf.float32, shape=[None])
    global_step = tf.Variable(0, trainable=False)

    # Construct the neural network, store the batch loss in a variable called `loss`.
    # At the end of this block, you'll want to have these ops:
    # - y_hat: probability of the positive class
    # - loss: the average cross entropy loss across the batch
    #   (hint: see tf.sigmoid_cross_entropy_with_logits)
    #   (hint 2: see tf.reduce_mean)
    # - train_op: the training operation resulting from minimizing the loss
    #             with a GradientDescentOptimizer
    # START YOUR CODE        
    # From Network to Logits
    with tf.name_scope('network_to_logits'):
        logits_ = fully_connected_layers(hidden_dims,x_ph)

    # Cost function from Logits
    with tf.name_scope('cost_function'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_, y_ph),name='loss')

    # Training Op  
    with tf.name_scope("training"):
        optimizer_ = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer_.minimize(loss)
    # Prediction
    with tf.name_scope("prediction"):
        y_hat = tf.nn.softmax(logits_, name="y_hat")
        
    # END YOUR CODE


    # Output some initial statistics.
    # You should see about a 0.6 initial loss (-ln 2).
    sess = tf.Session(config=tf.ConfigProto(device_filters="/cpu:0"))
    sess.run(tf.initialize_all_variables())
    print 'Initial loss:', sess.run(loss, feed_dict={x_ph: X, y_ph: y})

    for var in tf.trainable_variables():
        print 'Variable: ', var.name, var.get_shape()
        print 'dJ/dVar: ', sess.run(
                tf.gradients(loss, var), feed_dict={x_ph: X, y_ph: y})

    for epoch_num in xrange(num_epochs):
        for batch in xrange(0, X.shape[0], batch_size):
            X_batch = X[batch : batch + batch_size]
            y_batch = y[batch : batch + batch_size]

            # Populate loss_value with the loss this iteration.
            # START YOUR CODE
            loss_value = session.run(loss,
                       feed_dict={x_ph: X_batch, y_ph: y_batch})
            # END YOUR CODE
        if epoch_num % 300 == 0:
            print 'Step: ', global_step_value, 'Loss:', loss_value
            for var in tf.trainable_variables():
                print var.name, sess.run(var)
            print ''

    # Return your predictions.
    # START YOUR CODE
        
    preds = session.run(y_hat, feed_dict={x_ph: X, y_ph: y})
    
    return preds
    
    
    # END YOUR CODE
