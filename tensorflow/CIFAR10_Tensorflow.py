import tensorflow as tf
import numpy as np
import math
import timeit
from data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '.\\cifar-10'
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

def complex_model(X, y, is_training=True):
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
    W1 = tf.get_variable("W1", shape=[7, 7, 3, 32])
    b1 = tf.get_variable("b1", shape=[32])
    W2 = tf.get_variable("W2", shape=[1152, 1024])
    b2 = tf.get_variable("b2", shape=[1024])
    W3 = tf.get_variable("W3", shape=[1024, 10])
    b3 = tf.get_variable("b3", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, W1, strides=[1,2,2,1], padding='VALID') + b1
    h1 = tf.nn.relu(a1)
    print("after running ReLu")
    print(h1.shape)
    #h1 = tf.nn.batch_normalization(h1)
    h1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')
    print("after running maxpooling")
    print(h1.shape)
    h1_flat = tf.reshape(h1,[-1,1152])
    print("after reshape")
    
    a2 = tf.matmul(h1_flat,W2) + b2
    h2 = tf.nn.relu(a2)

    y_out = tf.matmul(h2, W3) + b3
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

    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    y_out = complex_model(X,y,is_training)

    # define our loss
    total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)

    # define our optimizer
    optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
    train_step = optimizer.minimize(mean_loss)

    def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
        tf.reset_default_graph()
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
        return total_loss,total_correct

    with tf.Session() as sess:
        with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
            sess.run(tf.global_variables_initializer())
            print('Training')
            run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
            print('Validation')
            run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

if __name__ == '__main__':
    main()