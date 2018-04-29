import numpy as np
import tensorflow as tf
from ops import masked_conv

def jacobi_itr(u_input, f_input, mask_1, mask_2):
    conductivity_1 = 16.
    conductivity_2 = 16.
    heat_filter_1 = conductivity_1 / 3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]).reshape(3,3,1,1)
    heat_filter_2 = conductivity_2 / 3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]).reshape(3,3,1,1)
    padded_input = tf.pad(u_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary

    # padded_mask_1 = tf.pad(u_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    # padded_mask_2 = tf.pad(u_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")  # convolution with symmetric padding at boundary
    # f_pred = masked_conv(padded_input, heat_filter_1, heat_filter_2, padded_mask_1, padded_mask_2)
    # diff = f_input[:,1:-1,1:-1,:] - f_pred[:,1:-1,1:-1,:]
    # d_matrix = mask_1*conductivity_1*(-8/3.) + mask_2*conductivity_2*(-8/3.)
    # u_new = diff/d_matrix[:,1:-1,1:-1,:]
    # u_new = tf.pad(u_new, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),"CONSTANT")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    f_pred = tf.nn.conv2d(input=padded_input, filter=heat_filter_1, strides=[1, 1, 1, 1], padding='VALID')
    diff = f_input[:,1:-1,1:-1,:] - f_pred[:,1:-1,1:-1,:]
    d_matrix = conductivity_1*(-8/3.)
    u_new = diff/d_matrix
    u_new = tf.pad(u_new, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),"CONSTANT")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    res = {}
    res['f_pred'] = f_pred
    res['d_matrix'] = d_matrix
    return u_new, res


def main():
    f_input = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    u_input = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    mask_1 = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    mask_2 = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    u_hist = [u_input]
    res_hist = []
    for i in range(400):
        u_new, res = jacobi_itr(u_hist[-1], f_input, mask_1, mask_2)
        u_hist += [u_new]
        res_hist += [res]


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

    import scipy.io as sio
    # u_ref = sio.loadmat('/home/hope-yao/Downloads/solution_6666.mat')['U1'][0]#[1:-1, 1:-1]
    # f_ref = sio.loadmat('/home/hope-yao/Downloads/q_6666.mat')['F1'][0]#[1:-1, 1:-1]
    f_ref = np.zeros((66, 66))
    f_ref[1:-1, 1:-1] = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0][1:-1, 1:-1]
    u_ref = np.zeros((66, 66))
    u_ref[1:-1, 1:-1] = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
    region_1 = np.asarray([[1.,]*33+[0.5]+[0.]*32]*66).reshape(1,66,66,1)
    region_2 = np.ones_like(region_1) - region_1
    loss_hist = []
    feed_dict = {f_input: f_ref.reshape(1, 66, 66, 1),
                 u_input: u_ref.reshape(1, 66, 66, 1),
                 mask_1: region_1,
                 mask_2: region_2}
    for ii in range(1, 400, 100):
        loss_hist += [sess.run(tf.reduce_mean(tf.abs(u_hist[ii] - u_ref)),feed_dict)]

    return u_hist

if __name__ == '__main__':
    cfg = {
        'batch_size': 1,
        'imsize': 64,
        'physics_problem': 'heat_transfer',  # candidates: 3D plate elasticity, helmholtz, vibro-acoustics
        'alpha': 5000,  # iteration
    }

    u_hist = main()
    print('done')