import numpy as np
import tensorflow as tf
from ops import new_weight_variable, new_bias_variable

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
            self.k = tf.Variable(1., tf.float32)
            lu_filter = 1/3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
            self.A_weights['LU_filter'] = np.reshape(lu_filter,(3,3,1,1)) * self.k
            lu_bias = np.zeros((self.batch_size, self.imsize, self.imsize, self.response_dim))
            self.A_weights['LU_bias'] = tf.Variable(lu_bias, dtype=tf.float32)
            self.A_weights['D_matrix'] = tf.tile(tf.reshape(-8.*self.k,(1,1,1,1)),(1,self.imsize,self.imsize,1))
        else:
            assert 'not supported'

    def LU_layers(self, input_tensor, LU_filter, LU_bias):
        return tf.nn.elu(tf.nn.conv2d(input=input_tensor, filter=LU_filter, strides=[1,1,1,1], padding='SAME') + LU_bias)

    def apply(self, f, u, max_itr=10):
        itr = 0
        result = {}
        result['u_hist'] = []
        while itr<max_itr:
            LU_u = self.LU_layers(u, self.A_weights['LU_filter'], self.A_weights['LU_bias'])
            u_new = (LU_u - f)  / self.A_weights['D_matrix']
            result['u_hist'] += [u_new]
            itr += 1
        return result



if __name__ == "__main__":
    from utils import creat_dir
    from tqdm import tqdm
    import scipy.io as sio

    cfg = {
            'batch_size': 16,
            'imsize': 64,
            'physics_problem': 'heat_transfer', # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
           }
    f = tf.placeholder(tf.float32,shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32,shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    jacobi = Jacobi_block(cfg)
    jacobi_result = jacobi.apply(f, u)

    # optimizer
    loss = tf.reduce_mean(tf.abs(jacobi_result['u_hist'][-1] - u ))
    lr = 0.0001
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate)#
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)

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
    y_train= np.tile(np.reshape(data_u,(1,cfg['imsize'],cfg['imsize'],1)),(128,1,1,1))
    x_train = np.zeros_like(y_train, dtype='float32')
    y_test= np.tile(np.reshape(data_u,(1,cfg['imsize'],cfg['imsize'],1)),(128,1,1,1))
    x_test = np.zeros_like(y_train, dtype='float32')


    batch_size = cfg['batch_size']
    it_per_ep = int( len(x_train) / batch_size )
    test_loss_hist = []
    train_loss_hist = []
    k_value_hist = []
    for itr in tqdm(range(50000)):
        for i in range(it_per_ep):
            x_input = x_train[i*batch_size : (i + 1)*batch_size]
            y_input = y_train[i*batch_size : (i + 1)*batch_size]
            feed_dict_train = {f: x_input, u: y_input}
            results = sess.run(train_op, feed_dict_train)

        if itr % 100 == 0:
            # monitor testing error
            for ii in range(int( len(x_test) / batch_size) ):
                x_input = x_test[ii * batch_size:(ii + 1) * batch_size]
                y_input = y_test[ii * batch_size:(ii + 1) * batch_size]
                feed_dict_test = {f: x_input, u: y_input}
                cur_test_loss = sess.run(loss, feed_dict_test)
                test_loss_hist += [cur_test_loss]
            # summary_test = sess.run(summary_op_test, feed_dict_test)
            # summary_writer.add_summary(summary_test, count)
            # monitor physical parameters
            k_value = sess.run(jacobi.k)
            k_value_hist += [k_value]
            # monitor training error
            for ii in range( int(len(x_train) / batch_size) ):
                x_input = x_train[ii * batch_size:(ii + 1) * batch_size]
                y_input = y_train[ii * batch_size:(ii + 1) * batch_size]
                feed_dict_train = {f: x_input, u: y_input}
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