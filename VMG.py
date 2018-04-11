import numpy as np
import tensorflow as tf
from Jacobi_block import Jacobi_block
import scipy.io as sio


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
        self.alpha_1 = cfg['alpha1']
        self.alpha_2 = cfg['alpha2']
        self.jacobi = jacobi
        self.cfg = cfg
        self.max_depth = self.cfg['max_depth']
        self.imsize = cfg['imsize']

    def Ax_net(self, input_tensor, jacobi):
        D_mat = -8. * self.jacobi.k
        LU_u = jacobi.LU_layers(input_tensor)
        return D_mat * input_tensor + LU_u

    def apply_MG_block(self, f, u, max_itr):
        result = {}
        u_hist = self.jacobi.apply(f, u, max_itr=max_itr)
        result['u'] = u_hist['final']
        ax = self.Ax_net(result['u'], self.jacobi)
        result['res'] = f-ax
        return result

    def apply(self, f, u):
        f_level = {}
        f_i = f
        f_level['1h'] = f
        for layer_i in range(1, self.max_depth, 1):
            f_level['{}h'.format(2**layer_i)] = f_i = tf.nn.avg_pool(f_i, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        u_h = {}
        r_h = {}
        cur_f = f
        # fine to coarse h, 2h, 4h, 8h, ...
        for layer_i in range(1,self.max_depth,1):
            mg_result = self.apply_MG_block(cur_f, cur_f, self.alpha_1)
            u_h['{}h'.format(2**(layer_i-1))] = mg_result['u']
            r_h['{}h'.format(2**(layer_i-1))] = mg_result['res']
            # downsample residual to next level input
            cur_f = tf.nn.avg_pool(mg_result['res'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # bottom level, lowest frequency part
        e_bottom = self.jacobi.apply(cur_f, cur_f, max_itr=self.alpha_2)
        u_h['{}h'.format(2 ** (self.max_depth - 1))] = e_bottom['final']

        u_pred = {}
        layer_i = 1
        # coarse to fine: ..., 8h, 4h, 2h, h
        u_pred['{}h'.format(2**(self.max_depth-1))] = cur_level_sol = u_h['{}h'.format(2**(self.max_depth-1))]
        while layer_i<self.max_depth: # 4h, 2h, h
            upper_level_sol = u_h['{}h'.format(2 ** (self.max_depth-layer_i-1))]
            upper_level_sol_dim = upper_level_sol.get_shape().as_list()[1:3]
            upsampled_cur_level_sol = tf.image.resize_images(cur_level_sol, size=upper_level_sol_dim,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            cur_level_sol = upper_level_sol + upsampled_cur_level_sol
            cur_level_sol_correct = self.jacobi.apply(f_level['{}h'.format(2**(self.max_depth-layer_i-1))], cur_level_sol, self.alpha_1)
            u_pred['{}h'.format(2**(self.max_depth-layer_i-1))] = cur_level_sol_correct['final']
            layer_i += 1
        u_pred['final'] = u_pred['1h']

        return u_pred

if __name__ == '__main__':
    from tqdm import tqdm

    cfg = {
        'batch_size': 16,
        'imsize': 68,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'max_depth': 3, # depth of V cycle, degrade to Jacobi if set to 0
        'alpha1': 10, # iteration at high frequency
        'alpha2': 100, # iteration at low freqeuncy
    }

    f = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize']-2, cfg['imsize'], 1))
    u = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize']-2, cfg['imsize'], 1))
    jacobi = Jacobi_block(cfg)
    vmg = VMG_geometric(cfg, jacobi)
    vmg_result = vmg.apply(f, f) #second f is inital guess for u

    # optimizer
    loss = tf.reduce_mean(tf.abs(vmg_result['final'] - u ))
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

    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    f1 = data['matrix'][0][0][1]
    A1 = data['matrix'][0][0][0]
    u1 = np.linalg.solve(A1, f1)
    u_gt = u1.reshape(1, cfg['imsize'] - 2, cfg['imsize'], 1)
    f1 = f1.reshape(1, cfg['imsize'] - 2, cfg['imsize'], 1)

    u_input = np.tile(u_gt, (16, 1, 1, 1))
    f_input = np.tile(f1, (16, 1, 1, 1))
    feed_dict_train = {f: f_input, u: u_input}
    loss_value, u_value = sess.run([loss, vmg_result['final']], feed_dict_train)

    import matplotlib.pyplot as plt

    plt.imshow(u_value[0, :, :, 0], cmap='hot')
    plt.grid('off')
    plt.colorbar()
    plt.show()