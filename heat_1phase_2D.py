import numpy as np
import scipy.io as sio
from scipy import signal

def load_data_elem():
    #u = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
    #f = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0][1:-1,1:-1]
    if  0:
        data = sio.loadmat('./data/heat_transfer/Heat_Transfer.mat')
        data_u = data['u']  # [data['u'][1+2*i,1+2*j] for i in range(32) for j in range(32)]
        u = np.reshape(data_u, (1, 64, 64, 1))
        f = np.zeros_like(u, dtype='float32')
        conductivity = np.float32(16.)

    data = sio.loadmat('/home/hope-yao/Downloads/matrix.mat')
    f = data['matrix'][0][0][1]
    A = data['matrix'][0][0][0]
    u = np.linalg.solve(A, f)
    u = u.reshape(66,68)
    f = f.reshape(66,68)
    conductivity = np.float32(16.)
    return u, f, conductivity

def masked_conv(node_resp, conductivity):

    x  = np.pad(node_resp, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")
    filter_val = conductivity * (1. / 3.)
    heat_filter = np.asarray([[filter_val, filter_val, filter_val],
                              [filter_val, 0, filter_val],
                              [filter_val, filter_val, filter_val]])
    conv_res = signal.correlate2d(x[0,:,:,0], heat_filter, mode='valid')
    return conv_res[2:-2,2:-2].reshape(1,64,64,1)

def apply(u_input, f, conductivity):
    padded_input = np.pad(u_input, ((1, 1), (1, 1)), "symmetric")
    filter_val = conductivity * (1. / 3.)
    heat_filter = np.asarray([[filter_val, filter_val, filter_val],
                              [filter_val, 0, filter_val],
                              [filter_val, filter_val, filter_val]])
    LU_u = signal.correlate2d(padded_input, heat_filter, mode='valid')
   # LU_u = signal.correlate2d(padded_input, filter=self.A_weights['LU_filter'], strides=[1, 1, 1, 1],padding='VALID')
    u = (f - LU_u[1:-1,:]) / (8*filter_val)
    result = np.pad(u, ((1, 1), (0, 0)), "constant")

    return result

def jacobi_itr(u_input, f_input, conductivity):

    LU_u = masked_conv(u_input, conductivity)
    d_matrix = conductivity * (-8. / 3.)
    u_new = (f_input - LU_u[:,:,1:-1,:]) / d_matrix
    u_new = np.pad(u_new[:,:,:,:], ((0, 0), (1, 1), (0, 0), (0, 0)),"constant")  # Dirc BC, MUST BE ENFORCED AT EVERY CONV ITERATION!
    return u_new, LU_u


def main():
    resp_gt, load_gt, conductivity = load_data_elem()

    n_itr = 30000
    u_hist = [np.zeros((68,68))]
    #LU_u_hist = []
    loss_hist = []
    for i in range(n_itr):
        #u_new, LU_u = jacobi_itr(u_hist[-1], load_gt, conductivity)
        u_new = apply(u_hist[-1], load_gt, conductivity)
        u_hist += [u_new]
        #LU_u_hist += [LU_u]
        loss_i = np.mean(np.abs(u_hist[i][1:-1,:] - resp_gt))
        loss_hist += [loss_i]
        print('n_itr: {}, loss: {}'.format(i,loss_i))


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_hist)
    plt.figure()
    plt.imshow(u_hist[-1][1:-1, 1:-1], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.figure()
    plt.imshow(resp_gt[1:-1, 1:-1], cmap='jet')
    plt.colorbar()
    plt.grid('off')
    plt.show()
    return u_hist

if __name__ == '__main__':
    u_hist = main()
