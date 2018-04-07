import numpy as np
import tensorflow as tf
from Jacobi_block import Jacobi_block
from VMG import VMG

def smoothing_restriction(img):
    '''
    http://www.maths.lth.se/na/courses/FMNN15/media/material/Chapter9.09c.pdf
    '''
    lp_filter = np.asarray([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]])
    lp_filter = tf.constant(lp_filter.reshape(3, 3, 1, 1), dtype=tf.float32)
    # use some image to double check on the filtering effect here
    smoothed_img = tf.nn.conv2d(input=img, filter=lp_filter, strides=[1, 1, 1, 1], padding='SAME')
    return smoothed_img


class FMG():
    def __init__(self,cfg):
        self.alpha_1 = 5
        self.alpha_2 = 50
        self.jacobi = Jacobi_block(cfg)
        self.max_depth = cfg['max_depth']

        self.vmg_stack = {}
        for depth_i in range(self.max_depth):
            cfg['max_depth'] = depth_i
            self.vmg_stack['depth_{}'.format(depth_i)] = VMG(cfg, self.jacobi)

    def apply(self, f, u):
        result = {}
        for depth_i in range(self.max_depth):
            result_i = self.vmg_stack['depth_{}'.format(depth_i)].apply(f,u)
            result['depth_{}'.format(depth_i)] = u = result_i['final']
        return result

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
    fmg = FMG(cfg)
    fmg_result = fmg.apply(f,u)

    # optimizer
    u_hat = fmg_result['layer_{}'.format(cfg['max_depth']-1)]
    loss = tf.reduce_mean(tf.abs(u_hat - u ))
    lr = 0.01
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)
