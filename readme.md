Todo:
1. Capable to solve FEM problems without training given known eq? Test with known para_k
2. how to handle boundary condition


observation for Jacobi(10):
1. Jacobi network will have very slow convergence at high resolution
2. Jacobi network with low resolution 8x8 can converge to ~6% training error but no more. And works under SGD, not Adadelta
3. Jacobi network with even lower resolution 4x4 can converge to <0.5% training error with SGD.lr=0.1 very fast
4. Jacobi network 4X4, training error<0.05%  with Adam.lr=0.01, <20epoch
5. Jacobi network 4X4, training error<0.01%  with Adam.lr=0.001, <10epoch
6. Jacobi network 4X4, training error<1e-5  with Adam.lr=0.0001, <20epoch
7. Jacobi network 4X4, converging slow again.  with Adam.lr=0.00001,

based on (6) above, use SGD.lr=0.0001 to train resolution 8x8 for Jacobi(10):
1. Jacobi network 8X8, training error<1e-5  with Adam.lr=0.0001, <20epoch
2. Jacobi network 16X16, training error<1e-5  with Adam.lr=0.0001, ~120epoch
3. Jacobi network 32x32, training error<1e-5  with Adam.lr=0.0001, ~130epoch
4. Jacobi network 64x64, training error<1e-5  with Adam.lr=0.0001, ~135epoch


3. the parameter k can't be successfully estimated in all cases.
4. in some cases, half of the predicted image has very small pixel value. Don't know why yet.