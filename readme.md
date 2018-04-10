
1. convolution with special boundary treatment is equvlent
![u_gt](./data/heat_transfer/ground_truth_u.png)
![u_conv](./data/heat_transfer/jacobi_wx_it612.png)


2. feed forward Jacobi network is converging, but slow
![Jacobi_forward_convergence](./data/heat_transfer/jacobi_wx_convergence.png)

3. parameter estimation requires very deep network
ground truth -> 16.0
400 layers -> 8.1
1500 layers -> 14.3
This is caused by the error in network prediction/ slow convergence v.s. network depth



