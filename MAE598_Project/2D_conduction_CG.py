import numpy as np
import scipy.io as sio

def conjgrad(A, b, tol, x):
    n = len(A)
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(5):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        #if np.sqrt(rsnew) < tol:
        print('Itr:', i)
        print(x)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

if __name__ == '__main__':
    #data1 = sio.loadmat('./data/heat_transfer_1phase/K_Forceboundary.mat')
    #data2 = sio.loadmat('./data/heat_transfer_1phase/f_forceboundary.mat')
    #A = data1['K_Forceboundary']
    #b = data2['f_forceboundary']

    # Toy matrix
    A = np.array([[4., 1.],[1., 3.]])
    b = np.array([[1.], [2.]])

    # 4x4 node matrix from K matlab code
    bb = np.array([[-0.295], [0], [0], [0.295], [-0.295], [0], [0], [0.295], [-0.295], [0], [0], [0.295], [-0.295], [0], [0], [0.295]])
    AA = np.array([[10.6666666666667, -2.66666666666667, 0, 0, -2.66666666666667, -5.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-2.66666666666667, 21.3333333333333, -2.66666666666667, 0, -5.33333333333333, -5.33333333333334, -5.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -2.66666666666667, 21.3333333333333, -2.66666666666666, 0, -5.33333333333333, -5.33333333333334, -5.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, -2.66666666666666, 10.6666666666667, 0, 0, -5.33333333333333, -2.66666666666667, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-2.66666666666667, -5.33333333333333, 0, 0, 21.3333333333333, -5.33333333333334, 0, 0, -2.66666666666667, -5.33333333333333, 0, 0, 0, 0, 0, 0],
                  [-5.33333333333333, -5.33333333333334, -5.33333333333333, 0, -5.33333333333334, 42.6666666666667, -5.33333333333333, 0, -5.33333333333333, -5.33333333333333, -5.33333333333333, 0, 0, 0, 0, 0],
                  [0, -5.33333333333333, -5.33333333333334, -5.33333333333333, 0, -5.33333333333333, 42.6666666666667, -5.33333333333333, 0, -5.33333333333333, -5.33333333333334, -5.33333333333333, 0, 0, 0, 0],
                  [0, 0, -5.33333333333333, -2.66666666666667, 0, 0, -5.33333333333333, 21.3333333333333, 0, 0, -5.33333333333333, -2.66666666666667, 0, 0, 0, 0],
                  [0, 0, 0, 0, -2.66666666666667, -5.33333333333333, 0, 0, 21.3333333333333, -5.33333333333334, 0, 0, -2.66666666666666, -5.33333333333333, 0, 0],
                  [0, 0, 0, 0, -5.33333333333333, -5.33333333333333, -5.33333333333333, 0, -5.33333333333334, 42.6666666666667, -5.33333333333334, 0, -5.33333333333333, -5.33333333333333, -5.33333333333333, 0],
                  [0, 0, 0, 0, 0, -5.33333333333333, -5.33333333333334, -5.33333333333333, 0, -5.33333333333334, 42.6666666666667, -5.33333333333333, 0, -5.33333333333333, -5.33333333333333, -5.33333333333333],
                  [0, 0, 0, 0, 0, 0, -5.33333333333333, -2.66666666666667, 0, 0, -5.33333333333333, 21.3333333333333, 0, 0, -5.33333333333333, -2.66666666666667],
                  [0, 0, 0, 0, 0, 0, 0, 0, -2.66666666666666, -5.33333333333333, 0, 0, 10.6666666666667, -2.66666666666667, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -5.33333333333333, -5.33333333333333, -5.33333333333333, 0, -2.66666666666667, 21.3333333333333, -2.66666666666667, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -5.33333333333333, -5.33333333333333, -5.33333333333333, 0, -2.66666666666667, 21.3333333333333, -2.66666666666667],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.33333333333333, -2.66666666666667, 0, 0, -2.66666666666667, 10.6666666666667]])

    tol = 1e-8
    print("A = ", A)
    print("b = ", b)
   # print("AA = ", AA)
   # print("bb = ", bb)

    #x = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    x = np.array([[0], [0]])
    x = conjgrad(A, b, tol, x)
    x_test = np.linalg.solve(A, b)

    # xx = conjgrad(AA, bb, tol)
    # xx_test = np.linalg.solve(AA, bb)
