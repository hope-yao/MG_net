import matplotlib.pyplot as plt
import numpy as np

def fmg_net_complexity():
    x=np.asarray([64,128,256,512,1024,2048,4096,8192])
    y=np.asarray([16,27,36,49,84,101,120,141])
    plt.semilogx(x*x,y,'o-')
    plt.xlabel('DOF')
    plt.ylabel('# of LU block in FMG-NET')