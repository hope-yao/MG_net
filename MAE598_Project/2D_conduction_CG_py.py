import numpy as np
import scipy.io as sio
from timeit import default_timer as timer

def conjgrad(A, b, tol, x):
    n = len(b)
    r = b - A.dot(x)
    #r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(n):
        Ap = A.dot(p)
        #Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            print('Itr:', i)
            #print(x)
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

if __name__ == '__main__':
    data1 = sio.loadmat('./data/100x100/K_Forceboundary_nodes100x100.mat')
    data2 = sio.loadmat('./data/100x100/f_forceboundary_nodes100x100.mat')
    data3 = sio.loadmat('./data/100x100/x0_nodes100x100.mat')
    A = data1['K_Forceboundary_nodes100x100']
    #AA = A.toarray()
    b = data2['f_forceboundary_nodes100x100']
    x = data3['x0_nodes100x100']

    # Toy Test Matrix A = 8x8
    b_test = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    A_test = np.array([[6, 0, 1, 2, 0, 0, 2, 1],
                  [0, 5, 1, 1, 0, 0, 3, 0],
                  [1, 1, 6, 1, 2, 0, 1, 2],
                  [2, 1, 1, 7, 1, 2, 1, 1],
                  [0, 0, 2, 1, 6, 0, 2, 1],
                  [0, 0, 0, 2, 0, 4, 1, 0],
                  [2, 3, 1, 1, 2, 1, 5, 1],
                  [1, 0, 2, 1, 1, 0, 1, 3]])
    x_test = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])

    # Tolerance: Decrease for grater accuracy
    tol = 1e-5

    start_py = timer()

    x_result_py = conjgrad(A, b, tol, x)

    end_py = timer()

    print('Python solved in ',  end_py - start_py, ' Seconds.')


    # Test Solutions

    #x_test = conjgrad(A, b, tol, x_test)
    #x_linalg = np.linalg.solve(AA, b)


