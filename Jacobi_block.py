import numpy as np
import tensorflow as tf
from ops import new_weight_variable, new_bias_variable

class Jacobi_block():
    def __init__(self,cfg):
        self.imsize = cfg['imsize']
        if cfg['physics_problem'] == 'heat_transfer':
            self.input_dim = 1
            self.response_dim = 1
            self.filter_size = 3
            self.A_weights = {}
            self.A_weights['LU_filter'] = new_weight_variable([self.filter_size, self.filter_size, self.input_dim, self.response_dim])
            self.A_weights['LU_bias'] = new_bias_variable([self.response_dim])
            self.A_weights['D_matrix'] = tf.Variable(np.ones((1, self.imsize, self.imsize, 1)), dtype=tf.float32, name='D_matrix')
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
    lr = 0.01
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
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

    import h5py
    x_train = np.zeros((16,64,64,1), dtype='float32')
    y_train= np.zeros((16,64,64,1), dtype='float32')
    x_test = np.zeros((16,64,64,1), dtype='float32')
    y_test = np.zeros((16,64,64,1), dtype='float32')

    count = 0
    batch_size = cfg['batch_size']
    it_per_ep = int( len(x_train) / batch_size )
    for itr in range(5000):
        for i in tqdm(range(it_per_ep)):
            x_input = x_train[i*batch_size : (i + 1)*batch_size]
            y_input = y_train[i*batch_size : (i + 1)*batch_size]
            feed_dict_train = {f: x_input, u: y_input}
            results = sess.run(train_op, feed_dict_train)

        test_loss = []
        train_loss = []
        if count % 100 == 0:
            # monitor testing error
            for ii in range(int( len(x_test) / batch_size) ):
                x_input = x_test[ii * batch_size:(ii + 1) * batch_size]
                y_input = y_test[ii * batch_size:(ii + 1) * batch_size]
                feed_dict_test = {f: x_input, u: y_input}
                test_loss += [sess.run(loss, feed_dict_test)]
            # summary_test = sess.run(summary_op_test, feed_dict_test)
            # summary_writer.add_summary(summary_test, count)
            # monitor training error
            for ii in range( int(len(x_train) / batch_size) ):
                x_input = x_train[ii * batch_size:(ii + 1) * batch_size]
                y_input = y_train[ii * batch_size:(ii + 1) * batch_size]
                feed_dict_train = {f: x_input, u: y_input}
                train_loss += [sess.run(loss, feed_dict_train)]
            # summary_train = sess.run(summary_op_train, feed_dict_train)
            # summary_writer.add_summary(summary_train, count)
            # summary_writer.flush()
            print("iter:{}  train_cost: {}  test_cost: {} ".format(count, np.mean(train_loss), np.mean(test_loss)))
            # if count % 50000 == 0:
            #     snapshot_name = "%s_%s" % ('experiment', str(count))
            #     fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
            #     print("Model saved in file: %s" % fn)
            #     sess.run(tf.assign(learning_rate, learning_rate * 0.5))
            # count += 1