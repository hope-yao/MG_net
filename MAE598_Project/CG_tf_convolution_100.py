import numpy as np
import scipy.io as sio
from timeit import default_timer as timer
import tensorflow as tf

def conjgrad_tf(A_weights, b, x, n):
    #r = b - A.dot(x)       # python method
    padded_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        # reshape
    A_dotx_conv = tf.nn.conv2d(input=padded_x, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')

    A_dotx_conv = tf.reshape(A_dotx_conv, (10100,1))
    r = b - A_dotx_conv
    p = r
    #rsold = np.dot(r.T, r)     # python method
    rsold = tf.matmul(tf.transpose(r), r)
    for i in range(n):
        #Ap = A.dot(p)      # python method
        padded_p = tf.pad(tf.reshape(p, (1, 100, 101, 1)), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        Ap_c = tf.nn.conv2d(input=padded_p, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')
        Ap = tf.reshape(Ap_c, (10100,1))
        # Ap = Ap_c[0, 0, :, :]
        #alpha = rsold / np.dot(p.T, Ap)        # python method
        alpha = rsold / tf.matmul(tf.transpose(p), Ap)
        x = tf.reshape(x, (10100, 1))
        x = x + alpha * p
        r = r - alpha * Ap
        #rsnew = np.dot(r.T, r)     # python method
        rsnew = tf.matmul(tf.transpose(r), r)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        #print('Itr:', i)
    return x


if __name__ == '__main__':
    # Filter
    filter = np.asarray([[-5.3333, -5.3333, -5.3333], [-5.3333, 42.6667, -5.3333], [-5.3333, -5.3333, -5.3333]])
    A_weights = np.reshape(filter, (3, 3, 1, 1))

    # 100 x 100 Element Data
    data1 = sio.loadmat('./data/100x100/K_forceboundary_elements100x100.mat')
    data2 = sio.loadmat('./data/100x100/f_forceboundary_elements100x100.mat')
    data3 = sio.loadmat('./data/100x100/x0_elements100x100.mat')
    A100 = data1['K_forceboundary_elements100x100']
    b100 = data2['f_forceboundary_elements100x100']
    x100 = data3['x0_elements100x100']
    b_tf100 = tf.convert_to_tensor(b100, dtype=tf.float32)
    x0_tf100 = np.zeros((1,10100,1,1), 'float32')
    x0_tf100 = np.reshape(x0_tf100, (1, 100, 101, 1))

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    #tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 100 x 100 Elements
    n100 = 313    # Based on # of python iterations
    start_tf100 = timer()
    x_result_tf100 = conjgrad_tf(A_weights, b_tf100, x0_tf100, n100)
    end_tf100 = timer()
    print('Tensorflow solved for 10 element case in ', end_tf100 - start_tf100, ' Seconds.')
