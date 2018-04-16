import numpy as np
import tensorflow as tf
from ops import new_weight_variable, new_bias_variable
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy as np

class Jacobi_block():
    def __init__(self,cfg):
        self.imsize = cfg['imsize']
        self.batch_size = cfg['batch_size']
        if cfg['physics_problem'] == 'heat_transfer':
            self.input_dim = 1
            self.response_dim = 1
            self.filter_size = 3
            self.A_weights = {}
            # self.A_weights['LU_filter'] = new_weight_variable([self.filter_size, self.filter_size, self.input_dim, self.response_dim])
            # self.A_weights['LU_bias'] = new_bias_variable([self.response_dim])
            # self.A_weights['D_matrix'] = tf.Variable(np.ones((1, self.imsize, self.imsize, 1)), dtype=tf.float32, name='D_matrix')

            # NOTICE: right now for homogeneous anisotropic material only!!
            self.k = tf.Variable(16., tf.float32)
            lu_filter = 1/3. * np.asarray([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
            self.A_weights['LU_filter'] = np.reshape(lu_filter,(3,3,1,1)) * self.k
            lu_bias = np.zeros((self.batch_size, self.imsize, self.imsize, self.response_dim))
            self.A_weights['LU_bias'] = tf.Variable(lu_bias, dtype=tf.float32)
            self.A_weights['D_matrix'] = -9./3.*self.k #tf.tile(tf.reshape(-8./3.*self.k,(1,1,1,1)),(1,self.imsize-2,self.imsize,1))
        else:
            assert 'not supported'

    def LU_layers(self, input_tensor):
        padded_input = tf.pad(input_tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        LU_u = tf.nn.conv2d(input=padded_input, filter=self.A_weights['LU_filter'], strides=[1, 1, 1, 1],padding='VALID')
        return LU_u

    def apply(self, f, u, max_itr=10):
        result = {}
        result['u_hist'] = [u]
        for itr in range(max_itr):
            u_input = tf.pad(u, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), "CONSTANT") # boundary condition
            LU_u = self.LU_layers(u_input)
            u = (f - LU_u[:, :, 1:-1]) / self.A_weights['D_matrix']
            result['u_hist'] += [u]
        result['final'] = u
        return result

if __name__ == "__main__":
    from utils import creat_dir
    from tqdm import tqdm

    cfg = {
            'batch_size': 1,
            'imsize': 64,
            'physics_problem': 'heat_transfer', # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
            'alpha': 3000,  # iteration
          }
    f = tf.placeholder(tf.float32,shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32,shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    jacobi = Jacobi_block(cfg)
    jacobi_result = jacobi.apply(f, f, max_itr=cfg['alpha'])# where u is initialed as f

    # optimizer
    jacobi_result['loss'] = loss = tf.reduce_mean(tf.abs(jacobi_result['final'] - u ))
    lr = 1.
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate)#
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    # f1 = data['matrix'][0][0][1]
    # A1 = data['matrix'][0][0][0]
    # u1 = np.linalg.solve(A1, f1)
    u1 = sio.loadmat('/home/hope-yao/Downloads/Solution_6664.mat')['U1'][:, 1:-1, :]
    f1 = sio.loadmat('/home/hope-yao/Downloads/Input_q.mat')['F1'][:, 1:-1, :]
    u_gt = u1.reshape(10,cfg['imsize'],cfg['imsize'],1)
    f1 = f1.reshape(10,cfg['imsize'],cfg['imsize'],1)

    batch_size = cfg['batch_size']
    test_loss_hist = []
    train_loss_hist = []
    k_value_hist = []
    for itr in tqdm(range(50000)):
        for i in range(1):
            u_input = u_gt
            f_input = f1
            feed_dict_train = {f: f_input, u: u_input}
            _, loss_value, k_value = sess.run([train_op, loss, jacobi.k], feed_dict_train)

            print("iter:{}  train_cost: {}  k_value: {}".format(itr, np.mean(loss_value), k_value))

    print('done')