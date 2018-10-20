import numpy as np
import scipy.io as sio
from scipy import signal

num_node = 66



def reference_jacobi_solver(A, f):

    D = np.diag(np.diag(A))
    LU = A-D
    u = np.zeros_like(f)
    u_hist = [u]
    LU_u_hist = []
    n_iter = 100
    for i in range(n_iter):
        LU_u = np.matmul(LU, u_hist[-1])
        u_new = np.linalg.solve(D, f-LU_u)
        u_hist += [u_new]
        LU_u_hist += [LU_u]

    u_hist = [u_hist[i].reshape(num_node, num_node).transpose((1,0)) for i in range(n_iter)]
    LU_u_hist = [LU_u_hist[i].reshape(num_node, num_node).transpose((1,0)) for i in range(n_iter)]
    res = {
        'u_hist' : u_hist,
        'LU_u_hist' : LU_u_hist
    }
    return res

def load_data_elem():
    '''loading data obtained from FEA simulation'''
    case = 3
    if case==0:
        # single phase testing
        data = sio.loadmat('./data/new_heat_transfer_multi_inclusion/0circle.mat')
        mask = np.ones((num_node-1,num_node-1))

    elif case==1:
        # different phase at each side
        data = sio.loadmat('./data/new_heat_transfer_multi_inclusion/2phase_simpletest.mat')
        mask = np.ones((num_node-1,num_node-1))
        for i in range(num_node-1):
            for j in range(num_node-1):
                if j>=5:
                    mask[i,j] = 0

    elif case==2:
        # single circle inclusion
        data = sio.loadmat('./data/new_heat_transfer_multi_inclusion/1circle_center_25_25_rad_17.mat')
        mask = np.ones((num_node-1,num_node-1))
        for i in range(num_node-1):
            for j in range(num_node-1):
                if (i-24)**2+(j-24)**2<17**2:
                    mask[i,j] = 0

    elif case==3:
        # 3 circles inclusion
        data = sio.loadmat('./data/new_heat_transfer_multi_inclusion/3circle_center_25_25_rad_17_center_55_55_rad_7_center_55_25_rad_7.mat')
        mask = np.ones((num_node-1,num_node-1))
        for i in range(num_node-1):
            for j in range(num_node-1):
                if (i-24)**2+(j-24)**2<17**2 or (i-24)**2+(j-54)**2<7**2 or (i-54)**2+(j-54)**2<7**2:
                    mask[i,j] = 0

    A = data['K']
    f = data['f']
    u = np.linalg.solve(A, f)
    u_img = u.reshape(num_node,num_node).transpose((1,0))
    f_img = f.reshape(num_node,num_node).transpose((1,0))

    #ref_res = reference_jacobi_solver(A,f)

    coef = {}
    coef['conductivity_1'] = np.float32(16.)
    coef['conductivity_2'] = np.float32(205.)
    coef['diag_coef_1'] = coef['conductivity_1'] * (1./3)
    coef['side_coef_1'] = coef['conductivity_1'] * (1./3)
    coef['diag_coef_2'] = coef['conductivity_2'] * (1./3)
    coef['side_coef_2'] = coef['conductivity_2'] * (1./3)

    from scipy import signal
    elem_mask = np.squeeze(mask)
    node_filter = np.asarray([[1 / 4.] * 2] * 2)
    node_mask_1 = signal.correlate2d(elem_mask, node_filter, boundary='symm')
    node_mask_2 = signal.correlate2d(np.ones_like(elem_mask) - elem_mask, node_filter, boundary='symm')

    d_matrix = (coef['conductivity_1'] *  node_mask_1 + coef['conductivity_2'] *  node_mask_2 ) * (-8./3)
    d_matrix[0,:] /=2
    d_matrix[-1,:] /=2
    d_matrix[:,0] /=2
    d_matrix[:,-1] /=2
    coef['d_matrix'] = d_matrix

    return u_img, f_img, mask, coef


def boundary_padding(x):
    ''' special symmetric boundary padding '''
    x = np.reshape(x,[1,x.shape[0],x.shape[1],1])
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


def masked_conv(elem_mask_orig, node_resp, coef):
    diag_coef_1, side_coef_1 = coef['diag_coef_1'], coef['diag_coef_1']
    diag_coef_2, side_coef_2 = coef['diag_coef_2'], coef['diag_coef_2']
    x = np.reshape(node_resp,[1,num_node+2,num_node+2,1])
    elem_mask = np.reshape(elem_mask_orig,[1,num_node+1,num_node+1,1])

    y_diag_1 = np.zeros((num_node,num_node))
    y_side_1 = np.zeros((num_node,num_node))
    y_diag_2 = np.zeros((num_node,num_node))
    y_side_2 = np.zeros((num_node,num_node))
    for i in range(1, x.shape[1]-1, 1):
        for j in range(1, x.shape[1]-1, 1):
            y_diag_1[i-1, j-1] = x[0, i-1, j-1, 0] * elem_mask[0, i-1, j-1, 0] *diag_coef_1 \
                                 + x[0, i-1, j+1, 0] * elem_mask[0, i-1, j, 0] *diag_coef_1 \
                                 + x[0, i+1, j-1, 0] * elem_mask[0, i, j-1, 0] *diag_coef_1 \
                                 + x[0, i+1, j+1, 0] * elem_mask[0, i, j, 0] *diag_coef_1
            y_side_1[i-1, j-1] = x[0, i-1, j, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i-1, j, 0]) / 2. *side_coef_1 \
                                 + x[0, i, j-1, 0] * (elem_mask[0, i-1, j-1, 0] + elem_mask[0, i, j-1, 0]) / 2. *side_coef_1\
                                 + x[0, i, j + 1, 0] * (elem_mask[0, i-1, j, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1\
                                 + x[0, i+1, j, 0] * (elem_mask[0, i, j-1, 0] + elem_mask[0, i, j, 0]) / 2. *side_coef_1
            y_diag_2[i-1, j-1] = x[0, i-1, j-1, 0] * (1-elem_mask[0, i-1, j-1, 0]) *diag_coef_2 \
                                 + x[0, i-1, j+1, 0] * (1-elem_mask[0, i-1, j, 0] )*diag_coef_2 \
                                 + x[0, i+1, j-1, 0] * (1-elem_mask[0, i, j-1, 0]) *diag_coef_2 \
                                 + x[0, i+1, j+1, 0] * (1-elem_mask[0, i, j, 0]) *diag_coef_2
            y_side_2[i-1, j-1] = x[0, i-1, j, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i-1, j, 0]) / 2. *side_coef_2 \
                                 + x[0, i, j-1, 0] * (2-elem_mask[0, i-1, j-1, 0] - elem_mask[0, i, j-1, 0]) / 2. *side_coef_2\
                                 + x[0, i, j + 1, 0] * (2-elem_mask[0, i-1, j, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2\
                                 + x[0, i+1, j, 0] * (2-elem_mask[0, i, j-1, 0] - elem_mask[0, i, j, 0]) / 2. *side_coef_2

    tmp = {
        'LU_u_1': y_diag_1 + y_side_1,
        'LU_u_2': y_diag_2 + y_side_2,
        'LU_u': y_diag_1 + y_side_1 + y_diag_2 + y_side_2
    }
    return tmp

def apply(u_input, f, mask, coef):
    '''jacobi iteration'''
    padded_input = boundary_padding(u_input) # for boundary consideration
    padded_mask = boundary_padding(mask) # for boundary consideration
    #LU_u = signal.correlate2d(padded_input, heat_filter, mode='valid') # perform convolution
    conv_res = masked_conv(padded_mask, padded_input, coef)
    LU_u = conv_res['LU_u']
    LU_u_bc = boundary_correct(LU_u)
    u = (f - LU_u_bc) / coef['d_matrix'] # jacobi formulation of linear system of equation solver
    return u

def save_gif(u_hist, data_dir):
    # data_dir = './3circles'
    import imageio
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    images = []
    for i in range(0, 80, 1):
        plt.figure()
        plt.imshow(u_hist[i], cmap='jet')
        plt.colorbar()
        plt.grid('off')
        filename = data_dir + '/itr_{}'.format(i)
        plt.title(filename)
        plt.savefig(filename)
        images.append(imageio.imread(filename + '.png'))
    imageio.mimsave(data_dir + '/jacobi.gif', images)
    plt.close('all')


def main():
    resp_gt, load_gt, mask, coef = load_data_elem()

    u_hist = [np.zeros((num_node,num_node))] 
    loss_hist = []
    for i in range(500000):
        u_new = apply(u_hist[-1], load_gt, mask, coef)
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
