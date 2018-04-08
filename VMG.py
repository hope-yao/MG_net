import numpy as np
import tensorflow as tf
from Jacobi_block import Jacobi_block

def smoothing_restriction(img):
    '''
    http://www.maths.lth.se/na/courses/FMNN15/media/material/Chapter9.09c.pdf
    '''
    lp_filter = np.asarray([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]])
    lp_filter = tf.constant(lp_filter.reshape(3, 3, 1, 1), dtype=tf.float32)
    # use some image to double check on the filtering effect here
    smoothed_img = tf.nn.conv2d(input=img, filter=lp_filter, strides=[1, 1, 1, 1], padding='SAME')
    return smoothed_img



class VMG_algebraic():
    def __init__(self, cfg, jacobi):
        self.alpha_1 = 5
        self.alpha_2 = 50
        self.jacobi = jacobi
        self.cfg = cfg
        self.max_depth = self.cfg['max_depth']

    def Ax_net(self, input_tensor, A_weights):
        LU_filter, LU_bias, D_mat = A_weights['LU_filter'], A_weights['LU_bias'], A_weights['D_matrix']
        LU_u = self.jacobi.LU_layers(input_tensor, LU_filter, LU_bias)
        return D_mat * input_tensor + LU_u

    def apply_MG_block(self,f,u):
        result = {}
        u_h_hist = self.jacobi.apply(f, u, max_itr=self.alpha_1)
        result['u_h'] = u_h_hist['u_hist'][-1]
        res = self.Ax_net(result['u_h'], self.jacobi.A_weights)
        result['res_pool'] = tf.nn.avg_pool(res, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        result['res_smooth'] = smoothing_restriction(res)
        return result

    def apply(self, f, u):
        if self.max_depth == 0:
            u_pred = {}
            sol = self.jacobi.apply(f, u, max_itr=self.alpha_2)
            u_pred['final'] = sol['u_hist'][-1]
            return u_pred

        u_h = {}
        r_h = {}
        # fine to coarse: 0,1,2,3
        for layer_i in range(self.max_depth):
            mg_result = self.apply_MG_block(f, u)
            f = mg_result['res_smooth'] # residual as input rhs
            u = tf.constant(np.zeros((self.cfg['batch_size'], self.cfg['imsize'], self.cfg['imsize'], 1), dtype='float32')) # all zero initial guess
            u_h['layer_{}'.format(layer_i)] = mg_result['u_h']
            r_h['layer_{}'.format(layer_i)] = mg_result['res_smooth']

        # bottom level, lowest frequency part
        e_bottom = self.jacobi.apply(mg_result['res_smooth'], u, max_itr=self.alpha_2)

        # coarse to fine: 3,2,1,0
        u_pred = {}
        for layer_i in range(self.max_depth - 1, -1, -1):
            if layer_i == self.max_depth - 1:
                u_pred['layer_{}'.format(layer_i)] = e_bottom['u_hist'][-1] + u_h['layer_{}'.format(layer_i)]
            else:
                u_pred['layer_{}'.format(layer_i)] = u_pred['layer_{}'.format(layer_i + 1)] + u_h[
                    'layer_{}'.format(layer_i)]
        u_pred['final'] = u_pred['layer_{}'.format(layer_i)]
        return u_pred



class VMG_geometric():
    def __init__(self, cfg, jacobi):
        self.alpha_1 = 10
        self.alpha_2 = 10
        self.jacobi = jacobi
        self.cfg = cfg
        self.max_depth = self.cfg['max_depth']

    def Ax_net(self, input_tensor, A_weights):
        LU_filter = A_weights['LU_filter']
        D_mat = -8. * self.jacobi.k * tf.ones_like(input_tensor)
        LU_bias = tf.zeros_like(input_tensor, dtype=tf.float32)
        LU_u = self.jacobi.LU_layers(input_tensor, LU_filter, LU_bias)
        return D_mat * input_tensor + LU_u

    def apply_MG_block(self,f,u):
        result = {}
        u_h_hist = self.jacobi.apply(f, u, max_itr=self.alpha_1)
        result['u_h'] = u_h_hist['u_hist'][-1]
        res = self.Ax_net(result['u_h'], self.jacobi.A_weights)
        result['res_pool'] = tf.nn.avg_pool(res, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        return result

    def apply(self, f, u):
        if self.max_depth == 0:
            u_pred = {}
            sol = self.jacobi.apply(f, u, max_itr=self.alpha_2)
            u_pred['final'] = sol['u_hist'][-1]
            return u_pred

        u_h = {}
        r_h = {}
        # fine to coarse: 0,1,2,3
        for layer_i in range(self.max_depth):
            mg_result = self.apply_MG_block(f, u)
            f = mg_result['res_pool'] # residual as input rhs
            u = tf.zeros_like(f) # all zero initial guess
            u_h['layer_{}'.format(layer_i)] = mg_result['u_h']
            r_h['layer_{}'.format(layer_i)] = mg_result['res_pool']

        # bottom level, lowest frequency part
        e_bottom = self.jacobi.apply(mg_result['res_pool'], u, max_itr=self.alpha_2)

        # coarse to fine: 3,2,1,0
        u_pred = {}
        for layer_i in range(self.max_depth - 1, -1, -1):
            if layer_i == self.max_depth - 1:
                coarse_img = e_bottom['u_hist'][-1]
            else:
                coarse_img = u_pred['layer_{}'.format(layer_i + 1)]
            fine_img = u_h['layer_{}'.format(layer_i)]
            fine_img_dim = fine_img.get_shape().as_list()[1:3]
            upsampled_coarse_img = tf.image.resize_images(coarse_img, size=fine_img_dim, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            u_pred['layer_{}'.format(layer_i)] = upsampled_coarse_img + fine_img

        u_pred['final'] = u_pred['layer_{}'.format(layer_i)]
        return u_pred

if __name__ == '__main__':
    from tqdm import tqdm
    import scipy.io as sio

    cfg = {
        'batch_size': 16,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'max_depth': 0, # depth of V cycle, degrade to Jacobi if set to 0
    }

    f = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    jacobi = Jacobi_block(cfg)
    # vmg = VMG_geometric(cfg, jacobi)
    # vmg_result = vmg.apply(f,u)

    vmg_result = jacobi.apply(f, u)


    # optimizer
    loss = tf.reduce_mean(tf.abs(vmg_result['final'] - u ))
    lr = 0.0001
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
            print("iter:{}  train_cost: {}  test_cost: {}  k_value: {}".format(itr, np.mean(cur_train_loss), np.mean(cur_test_loss), k_value))
    print('done')