import numpy as np
import scipy.io as sio
from scipy import signal

num_node = 65
omega = 2./3

def reference_jacobi_solver(A, f):

    D = np.diag(np.diag(A))
    inv_D = np.expand_dims(1. / np.diag(A), 1)
    LU = A-D
    u = np.zeros_like(f)
    u_hist = [u]
    u_img_hist = [u.reshape(num_node,num_node,2)]
    LU_u_img_hist = []
    n_iter = 100000
    for itr_i in range(n_iter):
        LU_u = np.matmul(LU, u_hist[-1])
        u_new = omega * inv_D * (f-LU_u) + (1-omega)*u_hist[-1]

        LU_u_x = np.asarray([LU_u[2 * i] for i in range(65 ** 2)]).reshape(65, 65).transpose((1, 0))
        LU_u_y = np.asarray([LU_u[2 * i + 1] for i in range(65 ** 2)]).reshape(65, 65).transpose((1, 0))
        LU_u_img = np.stack([LU_u_x, LU_u_y], -1)
        u_new_x = np.asarray([u_new[2 * i] for i in range(65 ** 2)]).reshape(65, 65).transpose((1, 0))
        u_new_y = np.asarray([u_new[2 * i + 1] for i in range(65 ** 2)]).reshape(65, 65).transpose((1, 0))
        u_new_img = np.stack([u_new_x, u_new_y], -1)
        # import matplotlib.pyplot as plt
        # plt.imshow(LU_u_x_hist)
        # plt.show()

        u_hist += [u_new]
        u_img_hist += [u_new_img]
        LU_u_img_hist += [LU_u_img]

    res = {
        'u_hist' : np.asarray(u_img_hist),
        'LU_u_hist' : np.asarray(LU_u_img_hist)
    }
    return res

def load_data_elem():
    '''loading data obtained from FEA simulation'''
    # linear elasticity, all steel, Yfix
    data = sio.loadmat('/home/hope-yao/Documents/MG_net/data/new_elasticity/block.mat')
    ux = data['d_x'].reshape(65,65).transpose((1,0)) #* 1e10
    uy = data['d_y'].reshape(65,65).transpose((1,0)) #* 1e10
    u_img = np.concatenate([np.expand_dims(ux, 2), np.expand_dims(uy, 2)], 2)
    fx = data['f_x'].reshape(65,65).transpose((1,0))
    fy = data['f_y'].reshape(65,65).transpose((1,0))
    f_img = -1. * np.concatenate([np.expand_dims(fx, 2), np.expand_dims(fy, 2)], 2)

    A = data['K']
    # f = np.asarray(zip(data['f_x'], data['f_y'])).flatten()
    f = np.asarray([var for tup in zip(data['f_x'], data['f_y']) for var in tup])

    if 0:
        # check FEA data is correct
        u = np.asarray([var for tup in zip(data['d_x'], data['d_y']) for var in tup])
        f_pred = np.matmul(A, u)
        import matplotlib.pyplot as plt
        f_pred_img = np.asarray([f_pred[2 * i] for i in range(65 ** 2)]).reshape(65, 65).transpose((1, 0))
        plt.imshow(f_pred_img)
        plt.show()
    # get reference jacobi data
    # ref_res = reference_jacobi_solver(A, f)

    coef = {
        'E': 20e10,  # 200e9,
        'mu': 0.25,
        'elem_mask': np.ones((num_node-1, num_node-1))
    }

    coef['wxx_1'], coef['wxy_1'], coef['wyx_1'], coef['wyy_1'] = get_w_matrix(coef)
    coef['wxx_2'], coef['wxy_2'], coef['wyx_2'], coef['wyy_2'] = get_w_matrix(coef)
    coef['d_matrix'] = np_get_D_matrix_elast(coef, mode='symm')

    return u_img, f_img, coef

def get_w_matrix(coef_dict):
    E, mu = coef_dict['E'], coef_dict['mu']
    cost_coef = E/16./(1-mu**2)
    wxx = cost_coef * np.asarray([
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
        [-8*(1 + mu / 3.),       32. * (1 - mu / 3.),       -8*(1 + mu / 3.)],
        [-4*(1 - mu / 3.),        16 * mu / 3.,             -4*(1 - mu / 3.)],
    ])

    wxy = wyx = cost_coef*4 * np.asarray([
        [2 * (mu + 1),        0,             -2 * (mu + 1)],
        [0,                        0,                  0],
        [-2 * (mu + 1),        0,             2 * (mu + 1)],
    ])

    wyy = cost_coef * np.asarray([
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
        [16 * mu / 3.,               32. * (1 - mu / 3.),        16 * mu / 3.],
        [-4 * (1 - mu / 3.),         -8 * (1 + mu / 3.),        -4 * (1 - mu / 3.)],
    ])
    return wxx, wxy, wyx, wyy

def np_get_D_matrix_elast(coef_dict, mode='symm'):
    # convolution with symmetric padding at boundary
    elem_mask = coef_dict['elem_mask']
    from scipy import signal
    elem_mask = np.squeeze(elem_mask)
    node_filter = np.asarray([[1 / 4.] * 2] * 2)
    # first material phase
    node_mask_1 = signal.correlate2d(elem_mask, node_filter, boundary=mode)#symm, fill
    # second material phase
    node_mask_2 = signal.correlate2d(np.ones_like(elem_mask) - elem_mask, node_filter, boundary=mode)

    wxx_1_center, wyy_1_center = coef_dict['wxx_1'][1,1], coef_dict['wyy_1'][1,1]
    wxx_2_center, wyy_2_center = coef_dict['wxx_2'][1,1], coef_dict['wyy_2'][1,1]

    d_matrix_xx = node_mask_1 * wxx_1_center + node_mask_2 * wxx_2_center
    d_matrix_yy = node_mask_1 * wyy_1_center + node_mask_2 * wyy_2_center
    d_matrix = np.concatenate([np.expand_dims(np.expand_dims(d_matrix_xx, 0), 3),
                               np.expand_dims(np.expand_dims(d_matrix_yy, 0), 3)
                               ],3)

    return d_matrix[0]


def boundary_padding(x):
    ''' special symmetric boundary padding '''
    x = np.expand_dims(x, 0)
    left = x[:, :, 1:2, :]
    right = x[:, :, -2:-1, :]
    upper = np.concatenate([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
    down = np.concatenate([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
    padded_x = np.concatenate([left, x, right], 2)
    padded_x = np.concatenate([upper, padded_x, down], 1)
    padded_x = padded_x[0,:,:,:]
    return padded_x

def boundary_correct(x):
    x[0,:] /=2
    x[-1,:] /=2
    x[:,0] /=2
    x[:,-1] /=2

    x = np.pad(x[1:-1,1:-1], ((1, 1), (1, 1), (0, 0)), "constant") # for boundary consideration
    return x


def np_faster_mask_conv_elast(node_resp, coef):
    node_resp = np.expand_dims(node_resp, 0)
    wxx_1, wxy_1, wyx_1, wyy_1 = coef['wxx_1'], coef['wxy_1'], coef['wyx_1'], coef['wyy_1'],
    wxx_2, wxy_2, wyx_2, wyy_2 = coef['wxx_2'], coef['wxy_2'], coef['wyx_2'], coef['wyy_2'],

    diag_coef_diff = wxx_1[0,0] - wxx_2[0,0]
    fist_side_coef_diff = wxx_1[1,0] - wxx_2[1,0]
    second_side_coef_diff = wxx_1[0,1] - wxx_2[0,1]
    diag_coef_2 = wxx_2[0,0]
    first_side_coef_2 = wxx_2[1,0]
    second_side_coef_2 = wxx_2[0,1]
    coupling_coef_diff = wxy_1[0,0] - wxy_2[0,0]
    coupling_coef_2 = wxy_2[0,0]

    node_resp_x = node_resp[:,:,:,:1]
    node_resp_y = node_resp[:,:,:,1:]
    zero_padded_resp_x = np.pad(node_resp_x, ((0, 0), (1, 1), (1, 1), (0, 0)), "constant")#constant
    zero_padded_resp_y = np.pad(node_resp_y, ((0, 0), (1, 1), (1, 1), (0, 0)), "constant")
    elem_mask = coef['elem_mask'].reshape(1,num_node-1, num_node-1, 1)
    #padded_resp_x = np.expand_dims(boundary_padding(node_resp_x[0]), 0)
    #padded_resp_y = np.expand_dims(boundary_padding(node_resp_y[0]), 0)
    #padded_elem_mask = np.expand_dims(boundary_padding(elem_mask[0]), 0)
    padded_resp_x = node_resp_x
    padded_resp_y = node_resp_y
    padded_elem_mask = elem_mask
    padded_mask = np.pad(padded_elem_mask, ((0, 0), (1, 1), (1, 1), (0, 0)), "symmetric")

    for i in range(1, padded_resp_x.shape[1]-1, 1):
        for j in range(1, padded_resp_x.shape[2]-1, 1):
            conv_result_x_i_j = \
            padded_mask[0, i - 1, j - 1, 0] * \
            (
                    padded_resp_x[0, i - 1, j - 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j - 1, 0] * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i - 1, j, 0] * second_side_coef_diff/ 2.
            ) + \
            padded_mask[0, i - 1, j, 0] * \
            (
                    padded_resp_x[0, i - 1, j + 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j + 1, 0]   * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i - 1, j, 0]   * second_side_coef_diff/ 2.
            ) + \
            padded_mask[0, i, j - 1, 0] * \
            (
                    padded_resp_x[0, i + 1, j - 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j - 1, 0]    * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i + 1, j, 0] * second_side_coef_diff/ 2.
            ) + \
            padded_mask[0, i, j, 0] * \
            (
                    padded_resp_x[0, i + 1, j + 1, 0] * diag_coef_diff
                    + padded_resp_x[0, i, j + 1, 0]   * fist_side_coef_diff/ 2.
                    + padded_resp_x[0, i + 1, j, 0] * second_side_coef_diff/ 2.
            ) + \
            coupling_coef_diff * \
            (
                    zero_padded_resp_y[0, i - 1, j - 1, 0] - zero_padded_resp_y[0, i - 1, j + 1, 0]
                    - zero_padded_resp_y[0, i + 1, j - 1, 0] + zero_padded_resp_y[0, i + 1, j + 1, 0]
            ) + \
            diag_coef_2 * \
            (
                    padded_resp_x[0, i + 1, j - 1, 0] + padded_resp_x[0, i + 1, j + 1, 0]
                    + padded_resp_x[0, i - 1, j - 1, 0] + padded_resp_x[0, i - 1, j + 1, 0]

            ) + \
            first_side_coef_2 * \
            (
                    padded_resp_x[0, i, j + 1, 0] + padded_resp_x[0, i, j - 1, 0]
            ) + \
            second_side_coef_2 * \
            (
                    padded_resp_x[0, i - 1, j, 0] + padded_resp_x[0, i + 1, j, 0]
            ) + \
            coupling_coef_2 * \
            (
                    zero_padded_resp_y[0, i - 1, j - 1, 0] - zero_padded_resp_y[0, i - 1, j + 1, 0]
                    - zero_padded_resp_y[0, i + 1, j - 1, 0] + zero_padded_resp_y[0, i + 1, j + 1, 0]
            )
            # response in x direction

            conv_result_x = np.reshape(conv_result_x_i_j, (1, 1)) if i == 1 and j == 1 \
                else np.concatenate([conv_result_x, np.reshape(conv_result_x_i_j, (1, 1))], axis=0)

            # response in y direction
            conv_result_y_i_j = \
                padded_mask[0, i - 1, j - 1, 0] * \
                (
                        padded_resp_y[0, i - 1, j - 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j - 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i - 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                padded_mask[0, i - 1, j, 0] * \
                (
                        padded_resp_y[0, i - 1, j + 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j + 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i - 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                padded_mask[0, i, j - 1, 0] * \
                (
                        padded_resp_y[0, i + 1, j - 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j - 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i + 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                padded_mask[0, i, j, 0] * \
                (
                        padded_resp_y[0, i + 1, j + 1, 0] * diag_coef_diff
                        + padded_resp_y[0, i, j + 1, 0] * second_side_coef_diff / 2.
                        + padded_resp_y[0, i + 1, j, 0] * fist_side_coef_diff / 2.
                ) + \
                coupling_coef_diff * \
                (
                        zero_padded_resp_x[0, i - 1, j - 1, 0] - zero_padded_resp_x[0, i - 1, j + 1, 0]
                        - zero_padded_resp_x[0, i + 1, j - 1, 0] + zero_padded_resp_x[0, i + 1, j + 1, 0]
                ) + \
                diag_coef_2 * \
                (
                        padded_resp_y[0, i + 1, j - 1, 0] + padded_resp_y[0, i + 1, j + 1, 0]
                        + padded_resp_y[0, i - 1, j - 1, 0] + padded_resp_y[0, i - 1, j + 1, 0]
                ) + \
                first_side_coef_2 * \
                (
                        padded_resp_y[0, i - 1, j, 0] + padded_resp_y[0, i + 1, j, 0]
                ) + \
                second_side_coef_2 * \
                (
                        padded_resp_y[0, i, j + 1, 0] + padded_resp_y[0, i, j - 1, 0]
                ) + \
                coupling_coef_2 * \
                (
                        zero_padded_resp_x[0, i - 1, j - 1, 0] - zero_padded_resp_x[0, i - 1, j + 1, 0]
                        - zero_padded_resp_x[0, i + 1, j - 1, 0] + zero_padded_resp_x[0, i + 1, j + 1, 0]
                )
            conv_result_y = np.reshape(conv_result_y_i_j, (1, 1)) if i == 1 and j == 1 \
                else np.concatenate([conv_result_y, np.reshape(conv_result_y_i_j, (1, 1))], axis=0)

    LU_u_x = np.reshape(conv_result_x, (1, num_node, num_node, 1))
    LU_u_y = np.reshape(conv_result_y, (1, num_node, num_node, 1))
    LU_u = np.concatenate([LU_u_x,LU_u_y],3)
    weight = np.ones_like(LU_u)
    # weight[:, 0, :, :] /= 2
    # weight[:, -1, :, :] /= 2
    # weight[:, :, 0, :] /= 2
    # weight[:, :, -1, :] /= 2
    tmp = {
        'LU_u': LU_u*weight,
    }
    return tmp


def apply(u_input, f, coef):
    '''jacobi iteration'''
    padded_input = boundary_padding(u_input) # for boundary consideration
    res = np_faster_mask_conv_elast(padded_input, coef)# perform convolution
    LU_u = res['LU_u'][0]
    LU_u_bc = boundary_correct(LU_u)
    u = omega * (f - LU_u_bc) / coef['d_matrix'] + (1-omega)*u_input# jacobi formulation of linear system of equation solver
    return u

def main():
    resp_gt, load_gt, coef = load_data_elem()

    u_hist = [np.zeros((num_node,num_node,2))]
    loss_hist = []
    for i in range(5000):
        u_new = apply(u_hist[-1], load_gt, coef)
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
