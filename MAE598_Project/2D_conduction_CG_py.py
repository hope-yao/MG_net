import numpy as np
import scipy.io as sio
from timeit import default_timer as timer

def conjgrad(A, b, tol, x):
    n = len(b)
    r = b - A.dot(x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(n):
        Ap = A.dot(p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            print('Itr:', i)
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

if __name__ == '__main__':

    # Tolerance: Decrease for grater accuracy
    tol = 1e-5

    # 10 x 10 Element Data
    data1 = sio.loadmat('./data/10x10/K_forceboundary_elements10x10.mat')
    data2 = sio.loadmat('./data/10x10/f_forceboundary_elements10x10.mat')
    data3 = sio.loadmat('./data/10x10/x0_elements10x10.mat')
    A10 = data1['K_forceboundary_elements10x10']
    b10 = data2['f_forceboundary_elements10x10']
    x10 = data3['x0_elements10x10']
    start_py10 = timer()
    x10_result_py = conjgrad(A10, b10, tol, x10)
    end_py10 = timer()
    print('Python solved for 10 element case in ',  end_py10 - start_py10, ' Seconds.')

    # 100 x 100 Element Data
    data4 = sio.loadmat('./data/100x100/K_forceboundary_elements100x100.mat')
    data5 = sio.loadmat('./data/100x100/f_forceboundary_elements100x100.mat')
    data6 = sio.loadmat('./data/100x100/x0_elements100x100.mat')
    A100 = data4['K_forceboundary_elements100x100']
    b100 = data5['f_forceboundary_elements100x100']
    x100 = data6['x0_elements100x100']
    start_py100 = timer()
    x100_result_py = conjgrad(A100, b100, tol, x100)
    end_py100 = timer()
    print('Python solved for 100 element case in ',  end_py100 - start_py100, ' Seconds.')

    # 1000 x 1000 Element Data
    data7 = sio.loadmat('./data/1000x1000/K_forceboundary_elements1000x1000.mat')
    data8 = sio.loadmat('./data/1000x1000/f_forceboundary_elements1000x1000.mat')
    data9 = sio.loadmat('./data/1000x1000/x0_elements1000x1000.mat')
    A1000 = data7['K_forceboundary_elements1000x1000']
    b1000 = data8['f_forceboundary_elements1000x1000']
    x1000 = data9['x0_elements1000x1000']
    start_py1000 = timer()
    x100_result_py = conjgrad(A1000, b1000, tol, x1000)
    end_py1000 = timer()
    print('Python solved for 1000 element case in ', end_py1000 - start_py1000, ' Seconds.')


    # # Debug/Test
    #
    # # Toy Test Matrix A = 8x8
    # b_test = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    # A_test = np.array([[6, 0, 1, 2, 0, 0, 2, 1],
    #                    [0, 5, 1, 1, 0, 0, 3, 0],
    #                    [1, 1, 6, 1, 2, 0, 1, 2],
    #                    [2, 1, 1, 7, 1, 2, 1, 1],
    #                    [0, 0, 2, 1, 6, 0, 2, 1],
    #                    [0, 0, 0, 2, 0, 4, 1, 0],
    #                    [2, 3, 1, 1, 2, 1, 5, 1],
    #                    [1, 0, 2, 1, 1, 0, 1, 3]])
    # x_test = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
    # x_test = conjgrad(A_test, b_test, tol, x_test)
    # x_linalg = np.linalg.solve(A_test, b_test)
    # print(x_test)
    # print(x_linalg)


