from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

import math


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # pass
        
        '''
        Xavier intilization??

        Looks like it's only the standard deviations.
        Weights should be initialized from a Gaussian 
        with standard deviation equal to weight_scale

        INPUT: (N, D)

        W1: (D, H)
        b1: (1, H)

        H: (N, H)

        W2: (H, C)
        b2: (1, C)

        OUTPUT: (N, C)
        '''
        #W1 = np.zeros((input_dim, hidden_dim))
        W1 = np.random.normal(0.0, scale=weight_scale, size = input_dim * hidden_dim)
        W1 = W1.reshape(input_dim, hidden_dim)
        b1 = np.zeros((1, hidden_dim))

        W2 = np.random.normal(0.0, scale=weight_scale, size=hidden_dim * num_classes)
        #W2 = np.zeros((hidden_dim, num_classes))
        W2 = W2.reshape((hidden_dim, num_classes))
        b2 = np.zeros((1, num_classes))
        
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # pass
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # hidden, (fc_cache, relu_cache) = affine_relu_forward(X, W1, b1)
        fc_out, fc_cache = affine_forward(X, W1, b1)

        relu_out, relu_cache = relu_forward(fc_out)
        
        scores, scores_cache = affine_forward(relu_out, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # pass
        reg = self.reg
        loss, df = softmax_loss(scores, y)
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # df: (N, C)
        # dW2: (H, C)
        # hidden: (N, H)

        # dW2 = relu_ouput.T.dot(df)
        df, dW2, db2 = affine_backward(df, scores_cache)
        # db2 = np.sum(df, axis=0)

        grads['W2'] = dW2 + reg * W2
        grads['b2'] = db2

        '''
        Original implementation, should use the functions defined previously

        # reverse process from upper layer
        # consider the relationship between delta2 and delta3
        # this is the backward of relu
        df = df.dot(W2.T)
        df = np.maximum(0, df)
        
        # dW1: (D, H)
        dW1 = X.T.dot(df)

        db1 = np.sum(df, axis=0)
        '''
        # consider the relationship between delta2 and delta3, backward of relu
        # df = df.dot(W2.T)
        df = relu_backward(df, relu_cache)

        df, dW1, db1 = affine_backward(df, fc_cache)
        
        grads['W1'] = dW1 + reg * W1
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # pass
        
        for i in range(self.num_layers):
            if i == 0:
                row = input_dim
            else:
                row = hidden_dims[i - 1]
            
            if i == self.num_layers - 1:
                col = num_classes
            else:
                col = hidden_dims[i]

            W = np.random.normal(loc=0.0, scale=weight_scale, size=row * col).reshape((row, col))
            
            b = np.zeros((1, col))

            idx = str(i + 1)

            self.params['W' + idx] = W
            self.params['b' + idx] = b 

            if self.use_batchnorm and i != self.num_layers - 1:
                gamma = np.ones((1, col))
                beta = b.copy()

                self.params['gamma' + idx] = gamma
                self.params['beta' + idx] = beta
        # for name in sorted(self.params):
        #     print(name, self.params[name].shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # pass
        # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        reg = self.reg
        reg_loss = 0
        self.cache = {}
        for i in range(self.num_layers):
            idx = str(i + 1)
            W = self.params['W' + idx]
            b = self.params['b' + idx]
            if self.use_batchnorm and i != self.num_layers - 1:
                gamma = self.params['gamma' + idx]
                beta = self.params['beta' + idx]
            reg_loss += 0.5 * reg * np.sum(W * W)
            
            cache = {}
            if i == 0:
                layer_out, layer_cache = affine_forward(X, W, b)
                cache['fc_cache'] = layer_cache
                if self.use_batchnorm:
                    layer_out, layer_cache = batchnorm_forward(layer_out, gamma, beta, self.bn_params[i])
                    cache['bn_cache'] = layer_cache
                layer_out, layer_cache = relu_forward(layer_out)
                cache['relu_cache'] = layer_cache
            elif i == self.num_layers - 1:
                layer_out, layer_cache = affine_forward(layer_out, W, b)
                cache['fc_cache'] = layer_cache
            else:
                layer_out, layer_cache = affine_forward(layer_out, W, b)
                cache['fc_cache'] = layer_cache
                if self.use_batchnorm:
                    layer_out, layer_cache = batchnorm_forward(layer_out, gamma, beta, self.bn_params[i])
                    cache['bn_cache'] = layer_cache
                layer_out, layer_cache = relu_forward(layer_out)
                cache['relu_cache'] = layer_cache
            self.cache['cache' + idx] = cache

        scores = layer_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # pass
        data_loss, df = softmax_loss(scores, y)
        loss = data_loss + reg_loss

        # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

        for i in reversed(range(self.num_layers)):
            idx = str(i + 1)
            cache = self.cache['cache' + idx]
            if i != self.num_layers - 1:
                df = relu_backward(df, cache['relu_cache'])
                if self.use_batchnorm:
                    df, dgamma, dbeta = batchnorm_backward(df, cache['bn_cache'])
                    grads['gamma' + idx] = dgamma
                    grads['beta' + idx] = dbeta
            df, dW, db = affine_backward(df, cache['fc_cache'])
            W = cache['fc_cache'][1]
            grads['W' + idx] = dW + reg * W
            grads['b' + idx] = db     

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
