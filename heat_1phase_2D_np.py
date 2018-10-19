import numpy as np
import scipy.io as sio
from scipy import signal
from timeit import default_timer as timer

start = timer()

def load_data_elem():
    '''loading data obtained from FEA simulation'''
    data = sio.loadmat('./data/heat_transfer_1phase/matrix1.mat')
    f = data['matrix1'][0][0][1]
    A = data['matrix1'][0][0][0]
    u = np.linalg.solve(A, f)
    u = u.reshape(66,68)
    f = f.reshape(66,68)
    conductivity = np.float32(16.)
    return u, f, conductivity

def apply(u_input, f, heat_filter, d_matrix):
    '''jacobi iteration'''
    padded_input = np.pad(u_input, ((1, 1), (1, 1)), "symmetric") # for boundary consideration
    LU_u = signal.correlate2d(padded_input, heat_filter, mode='valid') # perform convolution
    u = (f - LU_u[1:-1,:]) / d_matrix # jacobi formulation of linear system of equation solver
    result = np.pad(u, ((1, 1), (0, 0)), "constant") # for boundary consideration
    return result

def main():
    resp_gt, load_gt, conductivity = load_data_elem()
    filter_val = conductivity * (1. / 3.) # setting up filter element value
    heat_filter = np.asarray([[filter_val, filter_val, filter_val],
                              [filter_val, 0, filter_val],
                              [filter_val, filter_val, filter_val]]) # assemble filter
    d_matrix = -8*filter_val # value of d_matrix
    u_hist = [np.zeros((68,68))] # because the solution is 66x68, 2 extra column is left for boundary condition
    loss_hist = []
    for i in range(5000):
        u_new = apply(u_hist[-1], load_gt, heat_filter, d_matrix)
        u_hist += [u_new]
        loss_i = np.mean(np.abs(u_hist[i][1:-1,:] - resp_gt))
        loss_hist += [loss_i]
        print('n_itr: {}, loss: {}'.format(i,loss_i))

    end = timer()
    print(end - start) # Time in seconds
    print(u_hist)
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
