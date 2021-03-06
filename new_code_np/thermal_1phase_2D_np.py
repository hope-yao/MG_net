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
    R_u = signal.correlate2d(padded_input, heat_filter, mode='valid') # perform convolution
    R_u_bc = boundary_correct(R_u)
    u = (f - R_u_bc) / d_matrix # jacobi formulation of linear system of equation solver
    # u = 0.66 * (f - LU_u_bc) / d_matrix + (1 - 0.66) * u_input # worse convergence for laplacian
    return u


def visualize(loss_hist, resp_pred, resp_gt):
    import matplotlib.pyplot as plt
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.figure()
    plt.semilogy(loss_hist, 'b-', label='convergence')
    plt.semilogy([1, len(loss_hist)], [1, len(loss_hist) ** -1], 'k--', label='$O(n^{-1})$')
    plt.semilogy([1, len(loss_hist)], [1, len(loss_hist) ** -0.5], 'k--', label='$O(n^{-0.5})$')
    plt.legend()
    plt.xlabel('network depth')
    plt.ylabel('prediction error')
    BIGGER_SIZE = 12
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(resp_pred, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(1,3,2)
    plt.imshow(resp_gt, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.subplot(1,3,3)
    plt.imshow(resp_pred-resp_gt, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.grid('off')
    plt.show()


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
        loss_i = np.linalg.norm(u_hist[i] - resp_gt)/np.linalg.norm(resp_gt)
        loss_hist += [loss_i]
        print('n_itr: {}, loss: {}'.format(i,loss_i))

    visualize(loss_hist, u_hist[-1], resp_gt)
    return u_hist

if __name__ == '__main__':
    u_hist = main()
