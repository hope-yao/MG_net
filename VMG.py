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


class VMG():
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

if __name__ == '__main__':
    cfg = {
        'batch_size': 16,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'max_depth': 4, # depth of V cycle
    }

    f = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    u = tf.placeholder(tf.float32, shape=(cfg['batch_size'], cfg['imsize'], cfg['imsize'], 1))
    jacobi = Jacobi_block(cfg)
    vmg = VMG(cfg, jacobi)
    vmg_result = vmg.apply(f,u)

    # optimizer
    u_hat = vmg_result['final']
    loss = tf.reduce_mean(tf.abs(u_hat - u ))
    lr = 0.01
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)
