import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.sparse.linalg import spsolve

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
    result = {}
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
        result['final'] = x
    return result


if __name__ == '__main__':
    conductivity = tf.Variable(1., tf.float32)
    # Filter
    filter = 1/3. * np.asarray([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
    A_weights = np.reshape(filter, (3, 3, 1, 1)) * conductivity

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
    u = tf.placeholder(tf.float32,shape=(1, 110, 110, 1))
    x_result_tf10 = conjgrad_tf(A_weights, b_tf10, x0_tf10, n10)

    # optimizer
    x_result_tf10['loss'] = loss = tf.reduce_mean(tf.abs(x_result_tf10['final'] - u ))
    lr = 1
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate)#
    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)

    u1 = spsolve(A10, b10)
    u_gt = u1
    u_gt = u1.reshape(1, 110, 110, 1)
    b10 = b10.reshape(1, 110, 110, 1)
    f = tf.placeholder(tf.float32,shape=(1, 110, 110, 1))
    batch_size = 1
    test_loss_hist = []
    train_loss_hist = []
    k_value_hist = []
    for itr in range(500):
        for i in range(1):
            u_input = u_gt
            f_input = b10
            feed_dict_train = {f: f_input, u: u_input}
            _, loss_value, k_value = sess.run([train_op, loss, conductivity], feed_dict_train)

            print("iter:{}  train_cost: {}  k_value: {}".format(itr, np.mean(loss_value), k_value))

    print('done')


    # # 100 x 100 Elements
    # n100 = 313  # Based on # of python iterations
    # start_tf100 = timer()
    # x_result_tf100 = conjgrad_tf(A_weights, b_tf100, x0_tf100, n100)
    # end_tf100 = timer()
    # print('Tensorflow solved for 100 element case in ', end_tf100 - start_tf100, ' Seconds.')

    # # 1000 x 1000 Elements
    # n1000 =   # Based on # of python iterations
    # start_tf1000 = timer()
    # x_result_tf1000 = conjgrad_tf(A_weights, b_tf1000, x0_tf1000, n1000)
    # end_tf1000 = timer()
    # print('Tensorflow solved for 1000 element case in ', end_tf1000 - start_tf1000, ' Seconds.')


