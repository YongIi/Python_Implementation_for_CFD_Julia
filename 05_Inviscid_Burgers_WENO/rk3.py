# 开发人员：leo
# 开发时间：2023/4/18 11:13

# Time integeral using 3rd order Runge-Kutta numerical scheme

import numpy as np
from update_BC import update_BC
from get_LU import get_LU

def rk3(u,nt,dt,dx,BC_type):
    # un, ut, LU = np.zeros_like(u[0, :]), np.zeros_like(u[0, :]), np.zeros_like(u[0, :])
    ut = np.zeros_like(u[0, :])  # 主要是为了给边界赋值0
    un = u[0, :].copy()
    k = 0  # record index

    for i in range(1, nt + 1):

        # 3rd order Runge-Kutta numerical scheme
        # step 1
        LU = get_LU(dx, un)
        ut[3:-3] = un[3:-3] + dt * LU[3:-3]  # 前后2个ghost cells以及1个BC cell都不参与计算
        update_BC(ut, BC_type)
        # step 2
        LU = get_LU(dx, ut)
        ut[3:-3] = 0.75 * un[3:-3] + 0.25 * ut[3:-3] + 0.25 * dt * LU[3:-3]
        update_BC(ut, BC_type)
        # step 3
        LU = get_LU(dx, ut)
        un[3:-3] = (1.0 / 3.0) * un[3:-3] + (2.0 / 3.0) * ut[3:-3] + (2.0 / 3.0) * dt * LU[3:-3]
        update_BC(un, BC_type)

        k+=1
        u[k,:]=un
