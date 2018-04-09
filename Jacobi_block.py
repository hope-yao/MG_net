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

    D = A1.diag()
    D_inv = np.diag(1. / np.asarray(D))
    LU = A1 - np.diag(D)
    a, b = np.linalg.eig(np.matmul(D_inv, LU))

    u = np.zeros_like(f1)
    u_hist = [u]
    er_hist = [np.mean(np.abs(u_hist[-1] - u1))]
    for i in range(1000):
        u = np.matmul(D_inv, (f1 - np.matmul(LU, u)))
        u_hist += [u]
        er_hist += [np.mean(np.abs(u_hist[-1] - u1))]

    plt.plot(er_hist)
    plt.show()
    print('done')

def conv_vs_axpy():
    # axpy
    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    A1 = data['matrix'][0][0][0]
    f1 = data['matrix'][0][0][1]
    u1 = np.linalg.solve(A1, f1)
    b=np.matmul(A1,np.ones_like(f1))
    bb = b.reshape(66, 68)
    plt.figure()
    plt.imshow(bb)
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
    u[0,:,0,0] = 0
    u[0,:,-1,0] = 0
    u = tf.constant(u)
    output = tf.nn.conv2d(input=u, filter=w_filter, strides=[1,1,1,1], padding='SAME')
    img = sess.run(output)
    plt.figure()
    plt.imshow(img[0, :, 1:-1, 0])
    plt.colorbar()
    plt.show()
    print('done')
conv_vs_axpy()

def Jacobi_solver_conv():

    imsize = 64
    A_weights = {}
    A_weights['k'] = 16.
    lu_filter = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
    A_weights['LU_filter'] = np.reshape(lu_filter, (3, 3, 1, 1)) * A_weights['k']/ 3.
    A_weights['D_matrix'] = A_weights['k'] / 3. * tf.tile(tf.reshape(-8. * A_weights['k'], (1, 1, 1, 1)), (1, imsize, imsize, 1))

    data = sio.loadmat('./data/heat_transfer/Heat_Transfer.mat')
    # u_gt = np.asarray([data['u'][4+8*i,4+8*j] for i in range(8) for j in range(8)]).reshape(1,imsize,imsize,1)
    u_gt = data['u'].reshape(1,imsize,imsize,1)
    f = np.zeros_like(u_gt,dtype='float32')
    f [:,0,:,:] = 1.
    u = np.ones_like(f,dtype='float32') #initial guess
    # u[:,:,0,:] = 0
    # u[:,:,-1,:] = 0

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
    result['u_hist'] = [u]
    for itr in range(max_itr):
        LU_u = tf.nn.conv2d(input=u, filter=A_weights['LU_filter'], strides=[1,1,1,1], padding='SAME')
        u = sess.run((f - LU_u)  / A_weights['D_matrix'])
        result['u_hist'] += [u]
    plt.figure()
    plt.imshow(u_gt[0,:,:,0],cmap='hot')
    plt.colorbar()
    plt.figure()
    plt.imshow(result['u_hist'][-1][0,:,:,0],cmap='hot')
    plt.colorbar()
    plt.show()
    result['final'] = result['u_hist'][-1]
    return result
Jacobi_solver_conv()


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
            self.k = 16.#tf.Variable(1., tf.float32)
            lu_filter = 1/3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
            self.A_weights['LU_filter'] = np.reshape(lu_filter,(3,3,1,1)) * self.k
            lu_bias = np.zeros((self.batch_size, self.imsize, self.imsize, self.response_dim))
            self.A_weights['LU_bias'] = tf.Variable(lu_bias, dtype=tf.float32)
            self.A_weights['D_matrix'] = tf.tile(tf.reshape(-8.*self.k,(1,1,1,1)),(1,self.imsize,self.imsize,1))
        else:
            assert 'not supported'

    def LU_layers(self, input_tensor, LU_filter, LU_bias):
        return tf.nn.conv2d(input=input_tensor, filter=LU_filter, strides=[1,1,1,1], padding='SAME')
        # return tf.nn.elu(tf.nn.conv2d(input=input_tensor, filter=LU_filter, strides=[1,1,1,1], padding='SAME') + LU_bias)

    def apply(self, f, u, max_itr=10):
        itr = 0
        result = {}
        result['u_hist'] = []
        while itr<max_itr:
            LU_u = self.LU_layers(u, self.A_weights['LU_filter'], self.A_weights['LU_bias'])
            u_new = (f - LU_u)  / self.A_weights['D_matrix']
            result['u_hist'] += [u_new]
            u = u_new
            itr += 1
        result['final'] = u_new
        return result



if __name__ == "__main__":
    from utils import creat_dir
    from tqdm import tqdm

    cfg = {
            'batch_size': 16,
            'imsize': 64,
            'physics_problem': 'heat_transfer', # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
           }
    f = tf.placeholder(tf.float32,shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32,shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    jacobi = Jacobi_block(cfg)
    jacobi_result = jacobi.apply(f, u, max_itr=100)

    # optimizer
    # loss = tf.reduce_mean(tf.abs(jacobi_result['final'] - u ))
    # lr = 0.0001
    # learning_rate = tf.Variable(lr) # learning rate for optimizer
    # optimizer=tf.train.AdamOptimizer(learning_rate)#
    # grads=optimizer.compute_gradients(loss)
    # train_op=optimizer.apply_gradients(grads)

    # # monitor
    # logdir, modeldir = creat_dir("jacobi{}".format(lr))
    # summary_writer = tf.summary.FileWriter(logdir)
    # summary_op_train = tf.summary.merge([
    #     tf.summary.scalar("loss/loss_train", loss),
    #     tf.summary.scalar("lr/lr", learning_rate),
    # ])
    # summary_op_test = tf.summary.merge([
    #     tf.summary.scalar("loss/loss_test", loss),
    # ])
    # saver = tf.train.Saver()

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

    data = sio.loadmat('./data/heat_transfer/Heat_Transfer.mat')
    data_u = data['u']#[data['u'][1+2*i,1+2*j] for i in range(32) for j in range(32)]
    u_train= np.tile(np.reshape(data_u,(1,cfg['imsize'],cfg['imsize'],1)),(128,1,1,1))
    f_train = np.zeros_like(u_train, dtype='float32')
    u_test= np.tile(np.reshape(data_u,(1,cfg['imsize'],cfg['imsize'],1)),(128,1,1,1))
    f_test = np.zeros_like(u_test, dtype='float32')


    batch_size = cfg['batch_size']
    it_per_ep = int( len(f_train) / batch_size )
    test_loss_hist = []
    train_loss_hist = []
    k_value_hist = []
    for itr in tqdm(range(50000)):
        for i in range(it_per_ep):
            u_input = u_train[i*batch_size : (i + 1)*batch_size]
            f_input = f_train[i*batch_size : (i + 1)*batch_size]
            feed_dict_train = {f: f_input, u: u_input}
            results = sess.run(train_op, feed_dict_train)

        if itr % 100 == 0:
            # monitor testing error
            for ii in range(int( len(f_test) / batch_size) ):
                f_input = f_test[ii * batch_size:(ii + 1) * batch_size]
                u_input = u_test[ii * batch_size:(ii + 1) * batch_size]
                feed_dict_test = {f: f_input, u: u_input}
                cur_test_loss = sess.run(loss, feed_dict_test)
                test_loss_hist += [cur_test_loss]
            # summary_test = sess.run(summary_op_test, feed_dict_test)
            # summary_writer.add_summary(summary_test, count)
            # monitor physical parameters
            k_value = sess.run(jacobi.k)
            k_value_hist += [k_value]
            # monitor training error
            for ii in range( int(len(f_train) / batch_size) ):
                f_input = f_train[ii * batch_size:(ii + 1) * batch_size]
                u_input = u_train[ii * batch_size:(ii + 1) * batch_size]
                feed_dict_train = {f: f_input, u: u_input}
                cur_train_loss = sess.run(loss, feed_dict_train)
                train_loss_hist += [cur_train_loss]
            # summary_train = sess.run(summary_op_train, feed_dict_train)
            # summary_writer.add_summary(summary_train, count)
            # summary_writer.flush()
            print("iter:{}  train_cost: {}  test_cost: {}  k_value: {}".format(itr, np.mean(cur_train_loss), np.mean(cur_test_loss), k_value))
            # if count % 50000 == 0:
            #     snapshot_name = "%s_%s" % ('experiment', str(count))
            #     fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
            #     print("Model saved in file: %s" % fn)
            #     sess.run(tf.assign(learning_rate, learning_rate * 0.5))
            # count += 1

    print('done')