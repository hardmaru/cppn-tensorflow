'''
Implementation of Compositional Pattern Producing Networks in Tensorflow

https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

@hardmaru, 2016

'''

import numpy as np
import tensorflow as tf
from ops import *

class CPPN():
  def __init__(self, batch_size=1, z_dim = 32, c_dim = 1, scale = 8.0, net_size = 32):
    """

    Args:
    z_dim: how many dimensions of the latent space vector (R^z_dim)
    c_dim: 1 for mono, 3 for rgb.  dimension for output space.  you can modify code to do HSV rather than RGB.
    net_size: number of nodes for each fully connected layer of cppn
    scale: the bigger, the more zoomed out the picture becomes

    """

    self.batch_size = batch_size
    self.net_size = net_size
    x_dim = 256
    y_dim = 256
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.scale = scale
    self.c_dim = c_dim
    self.z_dim = z_dim

    # tf Graph batch of image (batch_size, height, width, depth)
    self.batch = tf.placeholder(tf.float32, [batch_size, x_dim, y_dim, c_dim])

    n_points = x_dim * y_dim
    self.n_points = n_points

    self.x_vec, self.y_vec, self.r_vec = self._coordinates(x_dim, y_dim, scale)

    # latent vector
    self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
    # inputs to cppn, like coordinates and radius from centre
    self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1])
    self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1])
    self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1])

    # builds the generator network
    self.G = self.generator(x_dim = self.x_dim, y_dim = self.y_dim)

    self.init()

  def init(self):

    # Initializing the tensor flow variables
    init = tf.initialize_all_variables()
    # Launch the session
    self.sess = tf.Session()
    self.sess.run(init)

  def reinit(self):
    init = tf.initialize_variables(tf.trainable_variables())
    self.sess.run(init)

  def _coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
    return x_mat, y_mat, r_mat

  def generator(self, x_dim, y_dim, reuse = False):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    net_size = self.net_size
    n_points = x_dim * y_dim

    # note that latent vector z is scaled to self.scale factor.
    z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
                    tf.ones([n_points, 1], dtype=tf.float32) * self.scale
    z_unroll = tf.reshape(z_scaled, [self.batch_size*n_points, self.z_dim])
    x_unroll = tf.reshape(self.x, [self.batch_size*n_points, 1])
    y_unroll = tf.reshape(self.y, [self.batch_size*n_points, 1])
    r_unroll = tf.reshape(self.r, [self.batch_size*n_points, 1])

    U = fully_connected(z_unroll, net_size, 'g_0_z') + \
        fully_connected(x_unroll, net_size, 'g_0_x', with_bias = False) + \
        fully_connected(y_unroll, net_size, 'g_0_y', with_bias = False) + \
        fully_connected(r_unroll, net_size, 'g_0_r', with_bias = False)


    '''
    Below are a bunch of examples of different CPPN configurations.
    Feel free to comment out and experiment!
    '''

    ###
    ### Example: 3 layers of tanh() layers, with net_size = 32 activations/layer
    ###
    #'''
    H = tf.nn.tanh(U)
    for i in range(3):
      H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_'+str(i)))
    output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
    #'''

    ###
    ### Similar to example above, but instead the output is
    ### a weird function rather than just the sigmoid
    '''
    H = tf.nn.tanh(U)
    for i in range(3):
      H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_'+str(i)))
    output = tf.sqrt(1.0-tf.abs(tf.tanh(fully_connected(H, self.c_dim, 'g_final'))))
    '''

    ###
    ### Example: mixing softplus and tanh layers, with net_size = 32 activations/layer
    ###
    '''
    H = tf.nn.tanh(U)
    H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_1'))
    H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_2'))
    H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_2'))
    H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_2'))
    H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_2'))
    output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
    '''

    ###
    ### Example: mixing sinusoids, tanh and multiple softplus layers
    ###
    '''
    H = tf.nn.tanh(U)
    H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_1'))
    H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_2'))
    H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_2'))
    output = 0.5 * tf.sin(fully_connected(H, self.c_dim, 'g_final')) + 0.5
    '''

    ###
    ### Example: residual network of 4 tanh() layers
    ###
    '''
    H = tf.nn.tanh(U)
    for i in range(3):
      H = H+tf.nn.tanh(fully_connected(H, net_size, g_tanh_'+str(i)))
    output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
    '''

    '''
    The final hidden later is pass thru a fully connected sigmoid later, so outputs -> (0, 1)
    Also, the output has a dimention of c_dim, so can be monotone or RGB
    '''
    result = tf.reshape(output, [self.batch_size, y_dim, x_dim, self.c_dim])

    return result

  def generate(self, z=None, x_dim = 26, y_dim = 26, scale = 8.0):
    """ Generate data by sampling from latent space.

    If z is not None, data for this point in latent space is
    generated. Otherwise, z is drawn from prior in latent
    space.
    """
    if z is None:
        z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution

    G = self.generator(x_dim = x_dim, y_dim = y_dim, reuse = True)
    x_vec, y_vec, r_vec = self._coordinates(x_dim, y_dim, scale = scale)
    image = self.sess.run(G, feed_dict={self.z: z, self.x: x_vec, self.y: y_vec, self.r: r_vec})
    return image

  def close(self):
    self.sess.close()
