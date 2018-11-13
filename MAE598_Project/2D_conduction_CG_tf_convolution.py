import numpy as np
import scipy.io as sio
from timeit import default_timer as timer
import tensorflow as tf

def conjgrad_py(A, b, tol, x):
    n = len(b)
    r = b - A.dot(x)
    # r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(n):
        Ap = A.dot(p)
        # Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            print('Itr:', i)
            print(x)
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def conjgrad_tf(A_weights, b, x, n):
    #r = b - A.dot(x)
    padded_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    r_c = b - tf.nn.conv2d(input=padded_x, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')
    r = r_c[0, 0, :, :]
    p = r_c
    #rsold = np.dot(r.T, r)
    rsold = tf.matmul(tf.transpose(r), r)
    for i in range(n):
        #Ap = A.dot(p)
        padded_p = tf.pad(p, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        Ap_c = tf.nn.conv2d(input=padded_p, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')
        Ap = Ap_c[0, 0, :, :]
        #alpha = rsold / np.dot(p.T, Ap)
        alpha = rsold / tf.matmul(tf.transpose(p[0, 0, :, :]), Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        #rsnew = np.dot(r.T, r)
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
    x0_tf10 = np.zeros((1,110,110,1), 'float32')

    # 100 x 100 Element Data
    data4 = sio.loadmat('./data/100x100/K_forceboundary_elements100x100.mat')
    data5 = sio.loadmat('./data/100x100/f_forceboundary_elements100x100.mat')
    data6 = sio.loadmat('./data/100x100/x0_elements100x100.mat')
    A100 = data4['K_forceboundary_elements100x100']
    b100 = data5['f_forceboundary_elements100x100']
    x100 = data6['x0_elements100x100']
    b_tf100 = tf.convert_to_tensor(b100, dtype=tf.float32)
    x0_tf100 = np.zeros((1, 10100, 10100, 1), 'float32')

    # # 1000 x 1000 Element Data
    # data7 = sio.loadmat('./data/1000x1000/K_forceboundary_elements1000x1000.mat')
    # data8 = sio.loadmat('./data/1000x1000/f_forceboundary_elements1000x1000.mat')
    # data9 = sio.loadmat('./data/1000x1000/x0_elements1000x1000.mat')
    # A1000 = data7['K_forceboundary_elements1000x1000']
    # b1000 = data8['f_forceboundary_elements1000x1000']
    # x1000 = data9['x0_elements1000x1000']
    # b_tf1000 = tf.convert_to_tensor(b1000, dtype=tf.float32)
    # x0_tf1000 = np.zeros((1, 1001000, 1001000, 1), 'float32')

    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    #tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 10 x 10 Elements
    n10 = 36    # Based on # of python iterations
    start_tf10 = timer()
    x_result_tf10 = conjgrad_tf(A_weights, b_tf10, x0_tf10, n10)
    end_tf10 = timer()
    print('Tensorflow solved for 10 element case in ', end_tf10 - start_tf10, ' Seconds.')

    # 100 x 100 Elements
    n100 = 313  # Based on # of python iterations
    start_tf100 = timer()
    x_result_tf100 = conjgrad_tf(A_weights, b_tf100, x0_tf100, n100)
    end_tf100 = timer()
    print('Tensorflow solved for 100 element case in ', end_tf100 - start_tf100, ' Seconds.')

    # # 1000 x 1000 Elements
    # n1000 =   # Based on # of python iterations
    # start_tf1000 = timer()
    # x_result_tf1000 = conjgrad_tf(A_weights, b_tf1000, x0_tf1000, n1000)
    # end_tf1000 = timer()
    # print('Tensorflow solved for 1000 element case in ', end_tf1000 - start_tf1000, ' Seconds.')

    # with tf.Session(config=tfconfig) as sess:
    #     sess.run(init)
    #
    #     start_tf = timer()
    #
    #     sess.run(conjgrad_tf(A_weights, b_tf, x0_tf))
    #
    #     end_tf = timer()
    #     print('Tensorflow solved in ',  end_tf - start_tf, ' Seconds.')

