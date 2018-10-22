import numpy as np
import scipy.io as sio
from tf_utils import *

class Jacobi_block():
    def __init__(self):
        # NOTICE: right now for homogeneous anisotropic material only!!
        self.rho = tf.Variable(16., tf.float32)
        R_filter = 1/3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
        self.R_filter = np.reshape(R_filter,(3,3,1,1)) * self.rho
        self.D_matrix  = tf.reshape(-8./3.*self.rho,(1,1,1,1))

        self.bc_mask = np.ones((batch_size, num_node, num_node, 1))
        self.bc_mask[:, 0, :, :] /= 2
        self.bc_mask[:, -1, :, :] /= 2
        self.bc_mask[:, :, 0, :] /= 2
        self.bc_mask[:, :, -1, :] /= 2

    def LU_layers(self, input_tensor):
        padded_input = boundary_padding(input_tensor)  # for boundary consideration
        # LU_u = signal.correlate2d(padded_input, heat_filter, mode='valid')  # perform convolution
        R_u = tf.nn.conv2d(input=padded_input, filter=self.R_filter, strides=[1, 1, 1, 1], padding='VALID')
        R_u_bc = R_u * self.bc_mask
        R_u_bc = tf.pad(R_u_bc[:, 1:-1, :, :], ((0,0), (1, 1), (0, 0), (0, 0)), "constant")  # for boundary consideration
        return R_u_bc


    def apply(self, f, max_itr=10):
        result = {}
        u_input = np.zeros((1, num_node, num_node, 1), 'float32')  # where u is unknown
        result['u_hist'] = [u_input]
        for itr in range(max_itr):
            R_u = self.LU_layers(result['u_hist'][-1])
            u = (f - R_u) / self.D_matrix  # jacobi formulation of linear system of equation solver
            result['u_hist'] += [u]

        result['final'] = result['u_hist'][-1]
        return result

def get_w_matrix(coef_dict):
    E, mu = coef_dict['E'], coef_dict['mu']
    cost_coef = E/16./(1-mu**2)
    wxx = cost_coef * np.asarray([
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
        [-8*(1 + mu / 3.),       32. * (1 - mu / 3.),       -8*(1 + mu / 3.)],
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
    ])

    wxy = wyx = cost_coef * np.asarray([
        [-2 * (mu + 1),        0,             2 * (mu + 1)],
        [0,                        0,                  0],
        [2 * (mu + 1),        0,             -2 * (mu + 1)],
    ])

    wyy = cost_coef * np.asarray([
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
        [16 * mu / 3.,               32. * (1 - mu / 3.),        16 * mu / 3.],
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
    ])
    return wxx, wxy, wyx, wyy

def np_get_D_matrix_elast(coef_dict, mode='symm'):
    # convolution with symmetric padding at boundary
    d_matrix_xx_val, d_matrix_yy_val = coef_dict['wxx'][1,1], coef_dict['wyy'][1,1]
    d_matrix_xx = d_matrix_xx_val*np.ones((num_node,num_node))
    d_matrix_yy = d_matrix_yy_val*np.ones((num_node,num_node))
    d_matrix = np.stack([d_matrix_xx, d_matrix_yy], -1)
    d_matrix[0,:] /=2
    d_matrix[-1,:] /=2
    d_matrix[:,0] /=2
    d_matrix[:,-1] /=2

    return d_matrix


def load_data_elem():
    '''loading data obtained from FEA simulation'''
    # linear elasticity, all steel, Yfix
    data = sio.loadmat('/home/hope-yao/Documents/MG_net/data/new_elasticity/crack.mat')
    ux = data['d_x'].reshape(65,65).transpose((1,0)) #* 1e10
    uy = data['d_y'].reshape(65,65).transpose((1,0)) #* 1e10
    u_img = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
    fx = data['f_x'].reshape(65,65).transpose((1,0))
    fy = data['f_y'].reshape(65,65).transpose((1,0))
    f_img = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2)

    A = data['K']
    # f = np.asarray(zip(data['f_x'], data['f_y'])).flatten()
    f = np.asarray([var for tup in zip(data['f_x'], data['f_y']) for var in tup])

    if 0:
        # check FEA data is correct
        u = np.asarray([var for tup in zip(data['d_x'], data['d_y']) for var in tup])
        f_pred = np.matmul(A, u)
        import matplotlib.pyplot as plt
        f_pred_img = np.asarray([f_pred[2 * i] for i in range(65 ** 2)]).reshape(65, 65).transpose((1, 0))
        plt.imshow(f_pred_img)
        plt.show()
    # get reference jacobi data
    #ref_res = reference_jacobi_solver(A, f)

    coef = {
        'E': 20e10,  # 200e9,
        'mu': 0.25,
    }

    coef['wxx'], coef['wxy'], coef['wyx'], coef['wyy'] = get_w_matrix(coef)
    coef['d_matrix'] = np_get_D_matrix_elast(coef, mode='symm')

    return u_img, f_img, coef

if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import creat_dir
    from tqdm import tqdm

    # build network
    batch_size = 1
    num_node = 66
    f = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1))
    u = tf.placeholder(tf.float32,shape=(batch_size, num_node, num_node, 1))
    jacobi = Jacobi_block()
    jacobi_result = jacobi.apply(f, max_itr=3000)

    # optimizer
    jacobi_result['loss'] = loss = tf.reduce_mean(tf.abs(jacobi_result['final'] - u ))
    lr = 1
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate)#
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)

    # initialize
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    u_img, f_img, conductivity = load_data_elem()
    loss_value_hist = []
    k_value_hist = []

    for itr in tqdm(range(10000)):
        for i in range(1):
            u_input = u_img
            f_input = f_img
            feed_dict_train = {f: f_input, u: u_input}
            _, loss_value_i, k_value_i = sess.run([train_op, loss, jacobi.rho], feed_dict_train)
            print("iter:{}  train_cost: {}  k_value: {}".format(itr, np.mean(loss_value_i), k_value_i))

    print('done')