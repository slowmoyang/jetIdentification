import torch

def maxout(x, m, k):
    '''
    ref. Ian J. Goodfellow et al. Maxout Networks.
    ref. Qi Wang, Joseph JaJa. From Maxout to Channel-Out: Encoding Information on Sparse Pathways.
    ref. http://www.simon-hohberg.de/2015/07/19/maxout.html
    m: Number of units in each linear feature extractor (complexity)
    k: Number of linear feature extractors
    '''
    d = x.get_shape().as_list()[-1] 
    with tf.variable_scope(name):

    W = torch.Tensor(
    b = tf.get_variable('b', shape=[m, k], initializer=tf.contrib.layers.xavier_initializer())
    # x: [N, d]
    # W: [d, m, k]
    # z: [N, m, k]
    // z = tf.tensordot(x, W, axes=[[1], [0]]) + b
    z = torch.mm(xx, ww).view(x.size(0), w.size(1), w.size(2))

    h = tf.reduce_max(z, axis=-1) # shape: [N, m]
    
    return h    
