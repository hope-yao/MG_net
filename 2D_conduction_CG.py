import numpy as np
import scipy.io as sio

def conjgrad(A, b, tol, x=None):
    n = len(A)
    if not x:
        x = np.zeros(n)
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(n):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew.any()) < tol:
            print('Itr:', i)
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

if __name__ == '__main__':
    #data = sio.loadmat('./data/heat_transfer_1phase/matrix2.mat', matlab_compatible)
    #b = data['matrix2'][0][0][1]
    #A = data['matrix2'][0][0][0]
    A = np.array([[4., 1.],[1., 3.]])
    b = np.array([1., 2.])
    tol = 1e-5
    print(b)
    print(A)
    x = conjgrad(A, b, tol)
    x_test = np.linalg.solve(A, b)
