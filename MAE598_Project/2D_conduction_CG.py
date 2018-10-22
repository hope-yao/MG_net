import numpy as np
import scipy.io as sio

def conjgrad(A, b, tol, x):
    n = len(A)
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(n):
        Ap = np.dot(A, p)
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

if __name__ == '__main__':
    #data1 = sio.loadmat('./data/K_Forceboundary_nodes4x4.mat')
    #data2 = sio.loadmat('./data/f_forceboundary_nodes4x4.mat')
    #data3 = sio.loadmat('./data/x0_nodes4x4.mat')
    #AA = data1['K_Forceboundary_nodes4x4']
    #Ab = data2['f_forceboundary_nodes4x4']
    #xx = data3['x0_nodes4x4']

    # Toy matrix A = 2x2
    #A = np.array([[4., 1.],[1., 3.]])
    #b = np.array([[1.], [2.]])

    # Toy matrix A = 8x8
    b = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    A = np.array([[6, 0, 1, 2, 0, 0, 2, 1],
                  [0, 5, 1, 1, 0, 0, 3, 0],
                  [1, 1, 6, 1, 2, 0, 1, 2],
                  [2, 1, 1, 7, 1, 2, 1, 1],
                  [0, 0, 2, 1, 6, 0, 2, 1],
                  [0, 0, 0, 2, 0, 4, 1, 0],
                  [2, 3, 1, 1, 2, 1, 5, 1],
                  [1, 0, 2, 1, 1, 0, 1, 3]])
    x = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
    tol = 1e-5
    x = conjgrad(A, b, tol, x)
    x_test = np.linalg.solve(A, b)


