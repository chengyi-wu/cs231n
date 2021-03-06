import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from data_utils import load_CIFAR10

import platform

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    system = platform.system()
    if system == "Darwin":
        cifar10_dir = './cifar-10-batches-py'
    elif system == 'Windows':
        cifar10_dir = '.\\cifar-10'
    else:
        cifar10_dir = './cifar-10'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

def simple_model(X,y):
    # define our weights (e.g. init_two_layer_convnet)
        
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_flat,W1) + b1
    return y_out

def complex_model(X,y,is_training):
    '''
    7x7 Convolutional Layer with 32 filters and stride of 1
    ReLU Activation Layer
    Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
    2x2 Max Pooling layer with a stride of 2
    Affine layer with 1024 output units
    ReLU Activation Layer
    Affine layer from 1024 input units to 10 outputs
    '''
    # define our weights (e.g. init_two_layer_convnet)
        
    # setup variables
    # 7x7x3 with 32 filters
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])

    # 32x32x3 -> conv -> 14x14x32 -> max_pool (2,2), strides = 2 -> 7 * 7 * 32 = 1152

    W1 = tf.get_variable("W1", shape=[16 * 16 * 32 , 1024])
    b1 = tf.get_variable("b1", shape=[1024])

    W2 = tf.get_variable("W2", shape=[1024 , 10])
    b2 = tf.get_variable("b2", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1 # 32*32*32
    h1 = tf.nn.relu(a1)
    h1 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 16*16*32

    h1_flat = tf.reshape(h1,[-1,16 * 16 * 32])
    a2 = tf.matmul(h1_flat,W1) + b1
    h2 = tf.nn.relu(a2)

    y_out = tf.matmul(h2, W2) + b2
    return y_out

def vgg_model(X, y, is_training):
    '''
    CONV3-64
    CONV3-128
    CONV3-256
    '''
    print("Running vgg_model")
    xavier = tf.contrib.layers.xavier_initializer()
    # CONV3-64
    Wconv1 = tf.get_variable(
        name='Wconv1',
        shape=(3,3,3,64),
        initializer=xavier
    )
    bconv1 = tf.get_variable(
        name='bconv1',
        shape=(64),
        initializer=tf.zeros_initializer
    )
    Wconv2 = tf.get_variable(
        name='Wconv2',
        shape=(3,3,64,64),
        initializer=xavier
    )
    bconv2 = tf.get_variable(
        name='bconv2',
        shape=(64),
        initializer=tf.zeros_initializer
    )

    # CONV3-128
    Wconv3 = tf.get_variable(
        name='Wconv3',
        shape=(3,3,64,128),
        initializer=xavier
    )
    bconv3 = tf.get_variable(
        name='bconv3',
        shape=(128),
        initializer=tf.zeros_initializer
    )
    Wconv4 = tf.get_variable(
        name='Wconv4',
        shape=(3,3,128,128),
        initializer=xavier
    )
    bconv4 = tf.get_variable(
        name='bconv4',
        shape=(128),
        initializer=tf.zeros_initializer
    )

    # CONV3-256
    Wconv5 = tf.get_variable(
        name='Wconv5',
        shape=(3,3,128,256),
        initializer=xavier
    )
    bconv5 = tf.get_variable(
        name='bconv5',
        shape=(256),
        initializer=tf.zeros_initializer
    )
    Wconv6 = tf.get_variable(
        name='Wconv6',
        shape=(3,3,256,256),
        initializer=xavier
    )
    bconv6 = tf.get_variable(
        name='bconv6',
        shape=(256),
        initializer=tf.zeros_initializer
    )
    Wconv7 = tf.get_variable(
        name='Wconv7',
        shape=(3,3,256,256),
        initializer=xavier
    )
    bconv7 = tf.get_variable(
        name='bconv7',
        shape=(256),
        initializer=tf.zeros_initializer
    )

    W1 = tf.get_variable(
        name="W1", 
        shape=[4 * 4 * 256, 10],
        initializer=xavier
    )
    b1 = tf.get_variable(
        name="b1",
        shape=[10],
        initializer=tf.zeros_initializer
    )

    # CONV RELU CONV RELU MAX_POOLING
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + bconv2
    h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 16x16x64

    a1 = tf.nn.conv2d(h1, Wconv3, strides=[1,1,1,1], padding='SAME') + bconv3
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv4, strides=[1,1,1,1], padding='SAME') + bconv4
    h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 8x8x128

    a1 = tf.nn.conv2d(h1, Wconv5, strides=[1,1,1,1], padding='SAME') + bconv5
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv6, strides=[1,1,1,1], padding='SAME') + bconv6
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv7, strides=[1,1,1,1], padding='SAME') + bconv7
    h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 4x4x256
    
    h1_flat = tf.reshape(h1,[-1, 4 * 4 * 256])
    y_out = tf.matmul(h1_flat,W1) + b1
    
    return y_out

def vgg_model2(X, y, is_training):
    '''
    (3,3) 32 filters
    '''
    print("Running vgg_model2")
    # 7x7x3 with 32 filters
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])

    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])

    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 64])
    bconv3 = tf.get_variable("bconv3", shape=[64])

    Wconv4 = tf.get_variable("Wconv4", shape=[7, 7, 32, 32])
    bconv4 = tf.get_variable("bconv4", shape=[32])

    Wconv5 = tf.get_variable("Wconv5", shape=[5, 5, 64, 64])
    bconv5 = tf.get_variable("bconv5", shape=[64])

    Wconv6 = tf.get_variable("Wconv6", shape=[3, 3, 64, 64])
    bconv6 = tf.get_variable("bconv6", shape=[64])

    W1 = tf.get_variable("W1", shape=[4*4*64,10])
    b1 = tf.get_variable("b1", shape=[10])

    # CONV RELU CONV RELU MAX_POOLING
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1
    a1 = tf.contrib.layers.batch_norm(a1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv4, strides=[1,1,1,1], padding='SAME') + bconv4
    a1 = tf.contrib.layers.batch_norm(a1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 16x16x32

    a1 = tf.nn.conv2d(h1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2
    a1 = tf.contrib.layers.batch_norm(a1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv5, strides=[1,1,1,1], padding='SAME') + bconv5
    a1 = tf.contrib.layers.batch_norm(a1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 8x8x64

    a1 = tf.nn.conv2d(h1, Wconv3, strides=[1,1,1,1], padding='SAME') + bconv3
    a1 = tf.contrib.layers.batch_norm(a1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.relu(a1)
    a1 = tf.nn.conv2d(h1, Wconv6, strides=[1,1,1,1], padding='SAME') + bconv6
    a1 = tf.contrib.layers.batch_norm(a1, center=True, scale=True, is_training=is_training)
    h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # 4x4x64
    
    h1_flat = tf.reshape(h1,[-1,4*4*64])
    y_out = tf.matmul(h1_flat,W1) + b1
    
    return y_out

def vgg_model3(X, y, is_training):
    '''
    CONV3-64
    CONV3-128
    CONV3-256
    '''
    print("Running vgg_model3")
    xavier = tf.contrib.layers.xavier_initializer()

    W1 = tf.get_variable(
        name="W1", 
        shape=[4 * 4 * 256, 10],
        initializer=xavier
    )
    b1 = tf.get_variable(
        name="b1",
        shape=[10],
        initializer=tf.zeros_initializer
    )

    # CONV RELU CONV RELU MAX_POOLING
    # CONV3-64
    a1 = tf.contrib.layers.conv2d(
        inputs=X,
        num_outputs=64,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.conv2d(
        inputs=h1,
        num_outputs=64,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    h1 = tf.contrib.layers.max_pool2d(
        inputs=h1, 
        kernel_size=2, 
        stride=2, 
        padding='VALID') # 16x16x64

    # CONV3-128
    a1 = tf.contrib.layers.conv2d(
        inputs=h1,
        num_outputs=128,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.conv2d(
        inputs=h1,
        num_outputs=128,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    h1 = tf.contrib.layers.max_pool2d(
        inputs=h1, 
        kernel_size=2, 
        stride=2, 
        padding='VALID') # 8x8x128

    # CONV3-256
    a1 = tf.contrib.layers.conv2d(
        inputs=h1,
        num_outputs=256,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.conv2d(
        inputs=h1,
        num_outputs=256,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.conv2d(
        inputs=h1,
        num_outputs=256,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None
    )
    a1 = tf.contrib.layers.batch_norm(
        inputs=a1,
        center=True,
        is_training=is_training
    )
    h1 = tf.nn.relu(a1)
    h1 = tf.contrib.layers.max_pool2d(
        inputs=h1, 
        kernel_size=2, 
        stride=2, 
        padding='VALID') # 4x4x256

    h1_flat = tf.reshape(h1,[-1, 4 * 4 * 256])
    y_out = tf.matmul(h1_flat,W1) + b1
    
    return y_out

def main():
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # clear old variables
    tf.reset_default_graph()

    # setup input (e.g. the data that changes every batch)
    # The first dim is None, and gets sets automatically based on batch size fed in
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # y_out = simple_model(X,y)
    y_out = vgg_model3(X,y,is_training=is_training)

    # define our loss
    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(y, 10), 
        logits=y_out
    )
    mean_loss = tf.reduce_mean(total_loss)

    # define our optimizer
    optimizer = tf.train.AdamOptimizer(1e-3) # select optimizer and set learning rate
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
    # optimizer = tf.train.GradientDescentOptimizer(1e-3)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)

    def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(predict,1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None
        
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [mean_loss,correct_prediction,accuracy]
        if training_now:
            variables[-1] = training
        
        # counter 
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
                
                # create a feed dictionary for this batch
                feed_dict = {X: Xd[idx,:],
                            y: yd[idx],
                            is_training: training_now }
                # get batch size
                actual_batch_size = yd[idx].shape[0]
                
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)
                
                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
                
                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                        .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                .format(total_loss,total_correct,e+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss,total_correct

    with tf.Session() as sess:
        with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0" 
            sess.run(tf.global_variables_initializer())
            print('Training')
            run_model(sess,y_out,mean_loss,X_train,y_train,5,64,100,train_step)
            print('Validation')
            run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

if __name__ == '__main__':
    main()