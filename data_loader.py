import numpy as np
import scipy.io as sio


def load_data_elem(case):
    # case = 1
    if case == -1:
        # toy case
        mask_1 =    np.asarray([[1,   1,   1,   1,   1,   1,   1,   1,   1,],
                                [1,   1,   1,   1,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   0,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   0,   0,   0,   0,]])
        mask_1 = np.asarray(mask_1, dtype='float32').reshape(1, 9, 9, 1)
        f = u = np.ones((1,10,10,1), dtype='float32')
        conductivity_1 = np.float32(10.)
        conductivity_2 = np.float32(100.)
    elif case == 0:
        mask_1 = sio.loadmat('./data/heat_transfer_2phase/mask.mat')['ind2']
        f = sio.loadmat('./data/heat_transfer_2phase/input_heatFlux.mat')['f1']
        u = sio.loadmat('./data/heat_transfer_2phase/steel_Aluminum_solution.mat')['u1']
        conductivity_1 = np.float32(16.)
        conductivity_2 = np.float32(205.)
    elif case == 1:
        mask_1 = sio.loadmat('./data/heat_transfer_2phase/mask.mat')['ind2']
        f = sio.loadmat('./data/heat_transfer_2phase/input_heatFlux.mat')['f1']
        u = sio.loadmat('./data/heat_transfer_2phase/steel_Air_solution.mat')['u1']
        conductivity_1 = np.float32(16.)
        conductivity_2 = np.float32(0.0262)

    return u, f, mask_1, conductivity_1, conductivity_2


def load_data(case):
    # case = 1
    if case == -1:
        # toy case
        mask_1 =    np.asarray([[1,   1,   1,   1,   1,   1,   1,   1,   1,   1,],
                                [1,   1,   1,   1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,],
                                [1,   1,   1,   1, 0.5,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1, 0.5,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1, 0.5,   0,   0,   0,   0,   0,],
                                [1,   1,   1,   1, 0.5, 0.5, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,],
                                [1,   1,   1,   1,   1,   1, 0.5,   0,   0,   0,]])
        mask_1 = np.asarray(mask_1, dtype='float32').reshape(1, 10, 10, 1)
        mask_2 = np.ones_like(mask_1, dtype='float32') - mask_1
        mask_2 = np.asarray(mask_2, dtype='float32').reshape(1, 10, 10, 1)
        f = u = np.ones_like(mask_1, dtype='float32')
        conductivity_1 = 10.
        conductivity_2 = 100.

    elif case == 0:
        # all steel
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
        f = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0]
        mask_1 = np.asarray([[1., ] * 33 + [0.5] * 1 + [0.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray([[0., ] * 33 + [0.5] * 1 + [1.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        conductivity_1 = 16.
        conductivity_2 = 16.
    elif case==1:
        # left: steel, right: aluminum
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/solution_6666.mat')['U1'][0][1:-1, 1:-1]
        f = sio.loadmat('/home/hope-yao/Downloads/q_6666.mat')['F1'][0]
        mask_1 = np.asarray([[1., ] * 33 + [0.5] * 1 + [0.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray([[0., ] * 33 + [0.5] * 1 + [1.] * 32] * 66, dtype='float32').reshape(1, 66, 66, 1)
        conductivity_1 = 16.
        conductivity_2 = 205.
    elif case==2:
        # center: aluminum, outer: steel
        u = np.zeros((1, 66, 66, 1), 'float32')
        u[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/solution_square.mat')['U1'][0][1:-1, 1:-1]
        f = np.zeros((1, 66, 66, 1), 'float32')
        f[0, 1:-1, 1:-1, 0] = sio.loadmat('/home/hope-yao/Downloads/q_square.mat')['F1'][0][1:-1, 1:-1]
        mask0 = sio.loadmat('/home/hope-yao/Downloads/index_square.mat')['B']
        mask = np.asarray(mask0, 'int')
        mask_1 = np.zeros((66, 66), dtype='float32')
        mask_2 = np.zeros((66, 66), dtype='float32')
        for i in range(66):
            for j in range(66):
                if mask[i, j] == 1:
                    mask_1[i, j] = 1
                if mask[i, j] == 0:
                    mask_1[i, j] = 0.5
                    mask_2[i, j] = 0.5
                if mask[i, j] == -1:
                    mask_2[i, j] = 1
        mask_1 = np.asarray(mask_1, dtype='float32').reshape(1, 66, 66, 1)
        mask_2 = np.asarray(mask_2, dtype='float32').reshape(1, 66, 66, 1)
        conductivity_1 = 16.
        conductivity_2 = 205.
    return u, f, mask_1, mask_2, conductivity_1, conductivity_2

