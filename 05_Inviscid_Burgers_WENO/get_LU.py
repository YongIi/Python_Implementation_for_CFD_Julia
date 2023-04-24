# 开发人员：leo
# 开发时间：2023/4/18 14:05

import numpy as np
from weno5_scheme import weno5L,weno5R

"""
LU = -udu/dx  但考虑了u[i]的速度方向，u[i]>0时用uL；u[i]<0时用uR
LU = -1/dx*(u^plus*(uLR-uLL)+u^minus*(uRR-uRL))
"""

def get_LU(dx, u):
    LU = np.zeros_like(u)
    """  # 该部分内容也放在weno5_scheme里面了
    uL = np.zeros_like(u)
    uR = np.zeros_like(u)

    for i in range(2,len(u)-3):
        uL[i] = weightsL(u[i-2], u[i-1], u[i], u[i+1], u[i+2])

    for i in range(3,len(u)-2):
        uR[i] = weightsR(u[i-2], u[i-1], u[i], u[i+1], u[i+2])
    """

    uL = weno5L(u)
    uR = weno5R(u)

    u_plus = get_u_plus(u)
    u_minus = get_u_minus(u)

    LU[3:-3]=-(u_plus[3:-3]*(uL[3:-3]-uL[2:-4])+u_minus[3:-3]*(uR[4:-2]-uR[3:-3]))/dx
    return LU

def get_u_plus(u):
    return np.maximum(u,0)

def get_u_minus(u):
    return np.minimum(u,0)

