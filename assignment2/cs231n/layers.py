from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # pass
    # row = x.shape[0]
    # out = []
    # for i in range(row):
    #     # flatten the input array to a vector
    #     # vector is 1 x D,
    #     # w is D x M
    #     # output will be 1 x M
    #     flat = x[i].flatten() 
    #     out.append(flat.dot(w) + b)
    # out = np.array(out)
    # out.reshape(row, - 1)
    N = x.shape[0]
    out = x.copy().reshape(N, -1).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # pass

    # dfdx = w # (D, M)
    # dfdw = x # (N, d1, ..., d_k)

    # local gradients
    dfdx = w.copy()
    dfdw = x.copy()
    dfdb = 1

    # global gradients received from upper layer 
    doutdf = dout.copy()

    # number of items
    N = doutdf.shape[0]

    # Why it's a sum??
    # chain rule, this should be the sum of all the changes
    db = np.sum(doutdf * dfdb, axis=0) 

    # dx = doutdf * dfdx
    # (N, M) * (D, M).T
    dx = doutdf.dot(dfdx.T).reshape(x.shape)
    
    # dw = doutdf * dfdw
    # (N, M) * (N, D)
    # dfdw.T * doutdf
    dfdw = dfdw.reshape(N, -1)
    dw = dfdw.T.dot(doutdf)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # pass
    out = x.copy()
    out = np.maximum(0, out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # pass

    ind = x.copy()
    ind = np.maximum(0, ind)
    ind[ind > 0] = 1

    dx = dout * ind

    # dx = dout.copy()
    # dx[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    # https://arxiv.org/pdf/1502.03167.pdf
    # code from the paper

    sample_mean = np.mean(x,axis=0)
    # sample_var = np.mean((x - sample_mean) ** 2, axis=0)
    sample_var = np.var(x, axis=0)

    # delta = x - sample_mean
    # sample_var = np.mean(delta ** 2, axis=0)

    # delta = x - sample_mean
    # sigma = sample_var + eps
    # sigmasqrt = np.sqrt(sigma)

    # x_head = delta / sigmasqrt

    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # pass
        '''
        update the running averages for mean and variance using
        an exponential decay based on the momentum parameter
        '''
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        x_head = (x - sample_mean) / np.sqrt(sample_var + eps)

        out = gamma * x_head + beta

        cache = (x, x_head, gamma, beta, eps, sample_mean, sample_var)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # pass
        
        # from the paper
        scale = gamma / np.sqrt(sample_var + eps)
        out = x * scale + beta - sample_mean * scale
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
 
    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var   

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # pass
    # x, x_head, gamma, beta, eps, sample_mean, sample_var, _, _, _ = cache
    x, x_head, gamma, beta, eps, sample_mean, sample_var = cache
    N, D = x.shape
    # running_mean = bn_param['running_mean']
    # running_var = bn_param['running_var']
    dgamma = np.sum(dout * x_head, axis=0)
    dbeta = np.sum(dout, axis=0)

    # refer to https://zhuanlan.zhihu.com/p/26138673 for details
    # refer to this: https://kevinzakka.github.io/2016/09/14/batch_normalization/
    # mu is a function of sigma, therefore it needs to add?

    doutdx_head = dout * gamma
    doutdvar = np.sum(doutdx_head * (x - sample_mean) * (-0.5) * ((sample_var + eps) ** - 1.5), axis=0)
    doutdmean = np.sum(- doutdx_head / np.sqrt(sample_var + eps), axis=0) # + doutdvar * np.sum(-2 * (x - sample_mean), axis=0) / N
    
    dx = doutdx_head / np.sqrt(sample_var + eps) + doutdvar * 2 * (x - sample_mean) / N + doutdmean / N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, x_head, gamma, beta, eps, sample_mean, sample_var = cache
    N, D = x.shape
    # running_mean = bn_param['running_mean']
    # running_var = bn_param['running_var']
    dgamma = np.sum(dout * x_head, axis=0)
    dbeta = np.sum(dout, axis=0)

    doutdx_head = dout * gamma

    delta = x - sample_mean
    sigma = sample_var + eps
    sigmasqrt = np.sqrt(sigma)

    doutdvar = np.sum(doutdx_head * delta * (-0.5) * (sigma ** - 1.5), axis=0)
    doutdmean = np.sum(- doutdx_head / sigmasqrt, axis=0) # + doutdvar * np.sum(-2 * delta, axis=0) / N
    
    dx = doutdx_head / sigmasqrt + doutdvar * 2 * delta / N + doutdmean / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # pass
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # pass
        '''
        If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
        '''
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # pass
        dx = dout.copy()
        dx *= mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # pass
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    # declare the output
    out = np.zeros((N, F, (H - HH + 2 * P) / S + 1, (W - WW + 2 * P) / S + 1))
    #print(out.shape)

    pad_width = ((0,0), (0,0), (P, P),(P, P))

    X = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    #print(x.shape, X.shape)

    for i in range(N):
        for f in range(F): # filters
            for h in range((H - HH + 2 * P) / S + 1): # row
                for j in range((W - WW + 2 * P) / S + 1): #col
                    out[i, f, h, j] += np.sum(X[i, :, h * S : h * S + HH, j * S : j * S + WW] * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # pass
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    db = np.zeros_like(b)

    for i in range(N):
        for f in range(F):
            db[f] += np.sum(dout[i, f, :, :])

    # dw is the smae shape as w (F, C, HH, WW)
    dw = np.zeros_like(w)
    # dx should has the same shape as x (N, C, H, W)
    dx = np.zeros_like(x)

    pad_width = ((0,0), (0,0), (P, P), (P, P))

    x = np.pad(x.copy(), pad_width=pad_width, mode='constant', constant_values=0)

    _, _, H_Prime, W_Prime = dout.shape

    for i in range(N):
        for f in range(F):
            for h in range(H_Prime):
                for j in range(W_Prime):
                    # consider it backward:
                    # X (3, 3) * W (3, 3) -> Y(1, 1)
                    # Y(1, 1) * X(3, 3) = W(3,3)
                    dw[f, :] += dout[i, f, h, j] * x[i, :, h * S : h * S + HH, j * S : j * S + WW]

    dx = np.zeros_like(x)

    # for i in range(N):
    #     for c in range(C):
    #         for f in range(F):
    #             for h in range(H_Prime): # row
    #                 for j in range(W_Prime): #col
    #                     # this should be the sum of the filters in each channel
    #                     '''
    #                     (3, 3) * (3, 3) => (1, 1)
    #                     The reverse should be (1, 1) * (3,3) = (3, 3)
    #                     Keep the dimension of the weights
    #                     '''
    #                     dx[i, c, h * S : h * S + HH, j * S : j * S + WW] += dout[i, f, h, j] * w[f, c, :, :]

    # turns out channel is optional
    for i in range(N):
        for f in range(F):
            for h in range(H_Prime): # row
                for j in range(W_Prime): #col
                    dx[i, :, h * S : h * S + HH, j * S : j * S + WW] += dout[i, f, h, j] * w[f]                    
    # Remove the zero padding
    dx = dx[:, :, P:-P, P:-P]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # pass
    
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    S = pool_param['stride']

    out = np.zeros((N, C, H // S, W // S))

    switch = np.zeros_like(out, dtype=int)

    for i in range(N):
        for c in range(C):
            for h in range(H // S):
                for w in range(W // S):
                    # pool is a (pool_height, pool_width matrix) matrix
                    pool = x[i, c, h * S : h * S + pool_height, w * S: w * S + pool_width]
                    idx = np.argmax(pool)
                    switch[i, c, h, w] = idx
                    out[i, c, h, w] = pool.flatten()[idx]

    #print(switch)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, switch)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    # pass
    x, pool_param, switch = cache
    
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    S = pool_param['stride']

    dx = np.zeros_like(x)

    for i in range(N):
        for c in range(C):
            for h in range(H // S):
                for w in range(W // S):
                    pool = dx[i, c, h * S : h * S + pool_height, w * S: w * S + pool_width]
                    pool = pool.flatten()
                    idx = switch[i, c, h, w]
                    pool[idx] = dout[i, c, h, w]
                    dx[i, c, h * S : h * S + pool_height, w * S: w * S + pool_width] = pool.reshape((pool_height, pool_width))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # pass
    _, C, H, W = x.shape
    out = np.zeros_like(x)
    cache = []
    for c in range(C):
        temp_bn, temp_cache = batchnorm_forward(x[:, c].reshape(-1, H * W), gamma[c], beta[c], bn_param)
        out[:, c] = temp_bn.reshape(-1, H, W)
        cache.append(temp_cache)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # pass
    _, C, H, W = dout.shape
    dx = np.zeros_like(dout)
    dgamma = np.zeros(C)
    dbeta = np.zeros(C)
    
    for c in range(C):
        df, gamma, beta \
            = batchnorm_backward(dout[:, c].reshape(-1, H * W), cache[c])
        dgamma[c] = np.sum(gamma)
        dbeta[c] = np.sum(beta)
        dx[:, c] = df.reshape(-1, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
