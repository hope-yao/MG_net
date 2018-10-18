import numpy as np
import scipy.io as sio
from scipy import signal

def load_data_elem():
    '''loading data obtained from FEA simulation'''
    # data = sio.loadmat('./data/heat_transfer_1phase/matrix.mat')
    # f = data['matrix'][0][0][1]
    # A = data['matrix'][0][0][0]

    # NEW MULTI CIRCLE CASE
    data = sio.loadmat('./data/new_heat_transfer_multi_inclusion/0circle.mat')
    A = data['K']
    f = data['f']
    u = np.linalg.solve(A, f)
    u_img = u.reshape(66,66).transpose((1,0))
    f_img = f.reshape(66,66).transpose((1,0))
    conductivity = np.float32(16.)
    return u_img, f_img, conductivity


def boundary_padding(x):
    ''' special symmetric boundary padding '''
    x = np.reshape(x,[1,66,66,1])
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = np.concatenate([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = np.concatenate([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = np.concatenate([left, x, right], 2)
    padded_x = np.concatenate([upper, padded_x, down], 1)
    padded_x = padded_x[0,:,:,0]
    return padded_x

def boundary_correct(x):
    x[0,:] /=2
    x[-1,:] /=2
    x[:,0] /=2
    x[:,-1] /=2

    x = np.pad(x[1:-1,:], ((1, 1), (0, 0)), "constant") # for boundary consideration
    return x

def apply(u_input, f, heat_filter, d_matrix):
    '''jacobi iteration'''
    padded_input = boundary_padding(u_input) # for boundary consideration
    LU_u = signal.correlate2d(padded_input, heat_filter, mode='valid') # perform convolution
    LU_u_bc = boundary_correct(LU_u)
    u = (f - LU_u_bc) / d_matrix # jacobi formulation of linear system of equation solver
    return u

def main():
    resp_gt, load_gt, conductivity = load_data_elem()
    filter_val = conductivity * (1. / 3.) # setting up filter element value
    heat_filter = np.asarray([[filter_val, filter_val, filter_val],
                              [filter_val, 0, filter_val],
                              [filter_val, filter_val, filter_val]]) # assemble filter
    d_matrix = -8*filter_val*np.ones_like(load_gt) # value of d_matrix
    d_matrix[0,:] /=2
    d_matrix[-1,:] /=2
    d_matrix[:,0] /=2
    d_matrix[:,-1] /=2

    u_hist = [np.zeros((66,66))] # because the solution is 66x68, 2 extra column is left for boundary condition
    loss_hist = []
    for i in range(5000):
        u_new = apply(u_hist[-1], load_gt, heat_filter, d_matrix)
        u_hist += [u_new]
        loss_i = np.mean(np.abs(u_hist[i] - resp_gt))
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
