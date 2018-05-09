import numpy as np
import tensorflow as tf
from ops import masked_conv, masked_conv_v2
from data_loader import load_data

def jacobi_itr(u_input, f_input, mask_1, mask_2, conductivity_1, conductivity_2):
    w_filter_1 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_1 / 3.
    w_filter_1 = tf.constant(w_filter_1.reshape((3, 3, 1, 1)))
    w_filter_2 = np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], 'float32') * conductivity_2 / 3.
    w_filter_2 = tf.constant(w_filter_2.reshape((3, 3, 1, 1)))

    conv_res = masked_conv(u_input, w_filter_1, w_filter_2, mask_1, mask_2, conductivity_1, conductivity_2)
    u_new = (f_input - conv_res['LU_u']) / conv_res['d_matrix']
    u_new = tf.pad(u_new[:,1:-1,1:-1,:], tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),"CONSTANT")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!

    return u_new, conv_res


def main():
    u_ref, f_ref, region_1, region_2, conductivity_1, conductivity_2 = load_data(case=1)
    f_input = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    u_input = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    mask_1 = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    mask_2 = tf.placeholder(tf.float32, shape=(1, 66, 66, 1))
    u_hist = [u_input]
    res_hist = []
    for i in range(2000):
        u_new, res = jacobi_itr(u_hist[-1], f_input, mask_1, mask_2, conductivity_1, conductivity_2)
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

    loss_hist = []
    feed_dict = {f_input: f_ref.reshape(1, 66, 66, 1),
                 u_input: np.zeros_like(u_ref.reshape(1, 66, 66, 1)),
                 mask_1: region_1,
                 mask_2: region_2}
    for ii in range(1, 2000, 100):
        loss_hist += [sess.run(tf.reduce_mean(tf.abs(u_hist[ii] - u_ref)),feed_dict)]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(sess.run(u_hist[-1][0, 1:-1, 1:-1, 0], feed_dict), cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.figure()
    plt.imshow(u_ref[0, 1:-1, 1:-1, 0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.show()
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