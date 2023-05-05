# 开发人员：leo
# 开发时间：2023/4/18 11:13

# Time integeral using 3rd order Runge-Kutta numerical scheme

import numpy as np
from update_BC import update_BC
from get_LU import get_LU

def rk3(gamma,q,ns,nt,dt,dx,BC_type):
    # numerical solsution at every time step
    qn = q[0, :, :].copy()  # qn[守恒量, 空间坐标]
    update_BC(qn, BC_type)
    # temporary array during RK3 integration
    qt = np.zeros_like(q[0, :, :])  # 主要是为了给边界赋值0
    # 空间维度
    # nx = len(q[0, 0, :])
    # print("空间维度：",nx)

    print("qt和qn的维度分别是：", qt.shape,qn.shape)

    freq = int(nt / ns)
    ri = 0  # record index

    for n in range(1, nt + 1):

        # 3rd order Runge-Kutta numerical scheme
        # step 1
        LU = get_LU(dx, qn, gamma)
        for i in range(3):
            qt[i,3:-3] = qn[i,3:-3] + dt * LU[i,3:-3]
        """  # 循环法，我还没测试循环法，但循环法里一般，网格遍历是外循环，3个守恒量是内循环
        for i in range(3, len(u) - 3):  # 从index=3开始，index=-4结束（包含2和-4）
            for k in range(3):
                qt[k, i] = qn[k, i] + dt * LU[k, i]
        """
        update_BC(qt, BC_type)

        # step 2
        LU = get_LU(dx, qt, gamma)
        for i in range(3):
            qt[i,3:-3] = 0.75 * qn[i,3:-3] + 0.25 * qt[i,3:-3] + 0.25 * dt * LU[i,3:-3]
        update_BC(qt, BC_type)

        # step 3
        LU = get_LU(dx, qt, gamma)
        for i in range(3):
            qn[i,3:-3] = (1.0 / 3.0) * qn[i,3:-3] + (2.0 / 3.0) * qt[i,3:-3] + (2.0 / 3.0) * dt * LU[i,3:-3]
        update_BC(qn, BC_type)

        if n % freq ==0:
            ri += 1
            q[ri,:,:] = qn
            print("结果输出时的ri:",ri)

        print("总时间步{0}，目前时间步{1}".format(nt, n))