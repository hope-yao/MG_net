import numpy as np
import scipy.io as sio
from timeit import default_timer as timer
import tensorflow as tf
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # set to -1 to enable CPU, set to 0 to enable GPU

def conjgrad_tf(A_weights, b, x, n):
    #r = b - A.dot(x)       # python method
    padded_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        # reshape
    A_dotx_conv = tf.nn.conv2d(input=padded_x, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')

    A_dotx_conv = tf.reshape(A_dotx_conv, (110,1))
    r = b - A_dotx_conv
    p = r
    #rsold = np.dot(r.T, r)     # python method
    rsold = tf.matmul(tf.transpose(r), r)
    for i in range(n):
        #Ap = A.dot(p)      # python method
        padded_p = tf.pad(tf.reshape(p, (1, 10, 11, 1)), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        Ap_c = tf.nn.conv2d(input=padded_p, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')
        Ap = tf.reshape(Ap_c, (110,1))
        # Ap = Ap_c[0, 0, :, :]
        #alpha = rsold / np.dot(p.T, Ap)        # python method
        alpha = rsold / tf.matmul(tf.transpose(p), Ap)
        x = tf.reshape(x, (110, 1))
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

    # 10 x 10 Element Data
    data1 = sio.loadmat('./data/10x10/K_forceboundary_elements10x10.mat')
    data2 = sio.loadmat('./data/10x10/f_forceboundary_elements10x10.mat')
    data3 = sio.loadmat('./data/10x10/x0_elements10x10.mat')
    A10 = data1['K_forceboundary_elements10x10']
    b10 = data2['f_forceboundary_elements10x10']
    x10 = data3['x0_elements10x10']
    b_tf10 = tf.convert_to_tensor(b10, dtype=tf.float32)
    x0_tf10 = np.zeros((1,110,1,1), 'float32')
    x0_tf10 = np.reshape(x0_tf10, (1, 10, 11, 1))

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 10 x 10 Elements
    n10 = 36    # Based on # of python iterations
    start_tf10 = timer()
    x_result_tf10 = conjgrad_tf(A_weights, b_tf10, x0_tf10, n10)
    end_tf10 = timer()
    print('Tensorflow solved for 10 element case in ', end_tf10 - start_tf10, ' Seconds.')
