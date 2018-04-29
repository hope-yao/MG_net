import tensorflow as tf
import numpy as np

def new_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def new_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def masked_conv(u, filter_1, filter_2, mask_1, mask_2, eps=0.01):
    '''
    convolution with two different kernels
    :param u: input image
    :param filter_1:
    :param filter_2:
    :param mask_1: region where filter_1 is applied, with value (0.5-eps,0.5+eps) on the boundary
    :param mask_2: region where filter_2 is applied, with value (0.5-eps,0.5+eps) on the boundary
    :param eps: some numerical consideration
    :return:
    '''
    output_1 = tf.nn.conv2d(input=u * tf.round(mask_1 + eps), filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')
    output_2 = tf.nn.conv2d(input=u * tf.round(mask_2 + eps), filter=filter_2, strides=[1, 1, 1, 1], padding='VALID')
    res1 = output_1 * mask_1[:,1:-1,1:-1,:] + output_2 * mask_2[:,1:-1,1:-1,:]

    output_1 = tf.nn.conv2d(input=u * mask_1, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')
    output_2 = tf.nn.conv2d(input=u * mask_2, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID')
    res2 = output_1 + output_2
    res = res1 * (tf.round(mask_1) + tf.round(mask_2))[:,1:-1,1:-1,:] + \
          res2 * (tf.ones_like(mask_1) - tf.round(mask_1) - tf.round(mask_2))[:,1:-1,1:-1,:]
    return res

if __name__ == '__main__':
    import scipy.io as sio
    import matplotlib.pyplot as plt
    case = 2
    if case==1:
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/solution_6666.mat')['U1'][0][1:-1, 1:-1]
        f = sio.loadmat('/home/hope-yao/Downloads/q_6666.mat')['F1'][0]
        mask_1 = np.asarray([[1., ] * 33 + [0.5] * 1 + [0.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray([[0., ] * 33 + [0.5] * 1 + [1.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
    elif case==2:
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/solution_square.mat')['U1'][0][1:-1, 1:-1]
        f = np.zeros((1, 66, 66, 1), 'float32')
        f[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/q_square.mat')['F1'][0][1:-1, 1:-1]
        mask0 = sio.loadmat('/home/hope-yao/Downloads/index_square.mat')['B']
        mask = np.asarray(mask0, 'int')
        mask_1 = np.zeros((66, 66), dtype='float32')
        mask_2 = np.zeros((66, 66), dtype='float32')
        for i in range(66):
            for j in range(66):
                if mask[i, j] == 1:
                    mask_1[i, j] = 1
                if mask[i, j] == 0:
                    mask_1[i, j] = 0.5
                    mask_2[i, j] = 0.5
                if mask[i, j] == -1:
                    mask_2[i, j] = 1
        mask_1 = np.asarray(mask_1, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray(mask_2, dtype='float32').reshape(1, 66, 66, 1)

    w_filter_1 = np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]], 'float32') * 16. / 3.
    w_filter_1 = tf.constant(w_filter_1.reshape((3, 3, 1, 1)))
    w_filter_2 = np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]], 'float32') * 205. / 3.
    w_filter_2 = tf.constant(w_filter_2.reshape((3, 3, 1, 1)))

    res = masked_conv(u, w_filter_1, w_filter_2, mask_1, mask_2)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    img = sess.run(res)
    # visualize convolution result
    plt.figure()
    plt.imshow(img[0, 1:-1, 1:-1, 0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    # visualize ground truth
    plt.figure()
    plt.imshow(f[0,1:-1, 1:-1,0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    # visualize error
    plt.figure()
    plt.imshow(f[1:-1, 1:-1]-img[0, 1:-1, 1:-1, 0], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.show()
    print('done')