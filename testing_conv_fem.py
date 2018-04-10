import numpy as np
import tensorflow as tf
from ops import new_weight_variable, new_bias_variable
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy as np


def Jacobi_solver_fem():
    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    A1 = data['matrix'][0][0][0]
    f1 = data['matrix'][0][0][1]
    u1 = np.linalg.solve(A1, f1)

    D = A1.diagonal()
    D_inv = np.diag(1. / np.asarray(D))
    LU = A1 - np.diag(D)
    a, b = np.linalg.eig(np.matmul(D_inv, LU))

    u = np.zeros_like(f1)
    u_hist = [u]
    er_hist = [np.mean(np.abs(u_hist[-1] - u1))]
    for i in range(200):
        u = np.matmul(D_inv, (f1 - np.matmul(LU, u)))
        u_hist += [u]
        er_hist += [np.mean(np.abs(u_hist[-1] - u1))]

    plt.plot(er_hist)
    plt.show()
    print('done')

def conv_vs_axpy():
    data = sio.loadmat('./data/heat_transfer/Heat_Transfer.mat')
    ftest = np.zeros((66,68),dtype='float32')
    ftest[1:65,2:66] = data['u']


    # axpy
    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    A1 = data['matrix'][0][0][0]
    f1 = data['matrix'][0][0][1]
    u1 = np.linalg.solve(A1, f1)
    b=np.matmul(A1,ftest.reshape(66*68,1))#np.ones_like(f1))
    bb = b.reshape(66, 68)
    plt.figure()
    plt.imshow(bb)
    plt.grid('off')
    plt.colorbar()

    # conv
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    # imsize = 64
    A_weights = {}
    A_weights['k'] = 16.
    w_filter = np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]],'float32') * A_weights['k'] / 3.
    w_filter = tf.constant(w_filter.reshape((3,3,1,1)))
    u = np.ones((1,68,68,1),'float32')
    u[0,1:-1,:,0] = ftest.reshape(66,68)
    u[0,0,:,0] = 0
    u[0,-1,:,0] = 0
    u = tf.constant(u)
    output = tf.nn.conv2d(input=u, filter=w_filter, strides=[1,1,1,1], padding='SAME')
    img = sess.run(output)
    plt.figure()
    plt.imshow(img[0,1:-1, :,  0])
    plt.colorbar()
    plt.grid('off')
    plt.show()
    print('error: {}'.format(np.mean(np.abs(bb-img[0,1:-1, :,  0]))))

def Jacobi_solver_conv():

    A_weights = {}
    A_weights['k'] = 16.
    lu_filter = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
    A_weights['LU_filter'] = np.reshape(lu_filter, (3, 3, 1, 1)) * A_weights['k']/ 3.
    A_weights['D_matrix'] = np.tile(np.reshape(-8. * A_weights['k']/3, (1, 1, 1, 1)), (1, 66, 68, 1))
    A_weights['D_matrix'] = tf.constant(A_weights['D_matrix'],dtype='float32')

    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    f1 = data['matrix'][0][0][1]
    A1 = data['matrix'][0][0][0]
    u1 = np.linalg.solve(A1, f1)
    u_gt = np.zeros((68,68))
    u_gt[1:-1,:] = u1.reshape(66,68)

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    max_itr = 1000
    itr = 0
    result = {}
    u_input = np.zeros((1, 68, 68, 1), 'float32')
    result['u_hist'] = [u_input]
    result['res_hist'] = [np.mean(np.abs(u_input-u_gt))]
    for itr in range(max_itr):
        LU_u = tf.nn.conv2d(input=tf.constant(u_input,dtype='float32'), filter=A_weights['LU_filter'], strides=[1,1,1,1], padding='SAME')
        u = sess.run((f1.reshape(1,66,68,1) - LU_u[:,1:-1,:,:])  / A_weights['D_matrix'])
        u_input = np.zeros((1, 68, 68, 1), 'float32')
        u_input[:, 1:-1, :, :] = u
        result['u_hist'] += [u_input]
        result['res_hist'] += [np.mean(np.abs(u_input-u_gt))]
    plt.figure()
    plt.imshow(u_gt[0,:,:,0],cmap='hot')
    plt.grid('off')
    plt.colorbar()
    plt.figure()
    plt.imshow(result['u_hist'][-1][0,1:-1,:,0],cmap='hot')
    plt.grid('off')
    plt.colorbar()
    plt.show()
    result['final'] = result['u_hist'][-1]
    return result


# Jacobi_solver_fem()
# conv_vs_axpy()
Jacobi_solver_conv()
