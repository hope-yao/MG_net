## convert to identity filter:
# 0*p11 + 0*p12 + 0*p13 + 0*p21 + 1*p22 + 1*p23 + 0*p31 + 1*p32 + -8*p33 = 0
# 0*p11 + 0*p12 + 0*p13 + 1*p21 + 1*p22 + 1*p23 + 1*p31 + -8*p32 + 1*p33 = 0
# 0*p11 + 0*p12 + 0*p13 + 1*p21 + 1*p22 + 0*p23 + -8*p31 + 1*p32 + 0*p33 = 0
#
# 0*p11 + 1*p12 + 1*p13 + 0*p21 + 1*p22 + -8*p23 + 0*p31 + 1*p32 + 1*p33 = 0
# 1*p11 + 1*p12 + 1*p13 + 1*p21 + -8*p22 + 1*p23 + 1*p31 + 1*p32 + 1*p33 = 0
# 1*p11 + 1*p12 + 0*p13 + -8*p21 + 1*p22 + 0*p23 + 1*p31 + 1*p32 + 0*p33 = 0
#
# 0*p11 + 1*p12 + -8*p13 + 0*p21 + 1*p22 + 1*p23 + 0*p31 + 0*p32 + 0*p33 = 0
# 1*p11 + -8*p12 + 1*p13 + 1*p21 + 1*p22 + 1*p23 + 0*p31 + 0*p32 + 0*p33 = 0
# -8*p11 + 1*p12 + 0*p13 + 1*p21 + 1*p22 + 0*p23 + 0*p31 + 0*p32 + 0*p33 = 0

from scipy import signal
A = [
[0, 0, 0, 0, 1, 1, 0, 1, -8],
[0, 0, 0, 1, 1, 1, 1, -8, 1],
[0, 0, 0, 1, 1, 0, -8, 1, 0],
[0, 1, 1, 0, 1, -8, 0, 1, 1],
[1, 1, 1, 1, -8, 1, 1, 1, 1],
[1, 1, 0, -8, 1, 0, 1, 1, 0],
[0, 1, -8, 0, 1, 1, 0, 0, 0],
[1, -8, 1, 1, 1, 1, 0, 0, 0],
[-8, 1, 0, 1, 1, 0, 0, 0, 0],]
w_identity = [0, 0, 0, 0, 1, 0, 0, 0, 0]
import numpy as np
w_precondition = np.linalg.solve(A, w_identity)
w_precondition = w_precondition.reshape(3,3)

w = np.asarray([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
signal.correlate2d(w, w_precondition, mode='same').tolist()


# dobule check:
import scipy.io as sio
u = sio.loadmat('/home/hope-yao/Downloads/steel_U.mat')['U1'][0][1:-1, 1:-1]
f = sio.loadmat('/home/hope-yao/Downloads/steel_q.mat')['F1'][0][1:-1,1:-1]
import matplotlib.pyplot as plt
plt.close()
f_precond = signal.correlate2d(f, w_precondition, mode='valid')
plt.figure()
plt.imshow(u)
plt.colorbar()
plt.figure()
plt.imshow(f)
plt.colorbar()
plt.figure()
plt.imshow(f_precond)
plt.colorbar()
plt.show()
