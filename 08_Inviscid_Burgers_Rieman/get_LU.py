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

    uL = weno5L(u)  # construct left state at i+1/2
    uR = weno5R(u)  # construct right state at i-1/2

    # Computing fluxes
    fL = get_fluxes(uL)
    fR = get_fluxes(uR)

    # compute Rusanov Riemann solver (flux at interface i-1/2)
    f = rusanov(fL, fR, u, uL, uR)

    # compute LU using Rusanov Riemann solver
    LU[3:-3] = -(f[4:-2]-f[3:-3])/dx
    return LU

# propagation speed
def get_wavespeed(u):
    c = np.zeros_like(u)
    for i in range(2,len(u)-2):  # 从index=2开始，index=-3结束（包含2和-3）
        c[i] = max(abs(u[i-2]),abs(u[i-1]),abs(u[i]),abs(u[i+1]),abs(u[i+2]))
        # c[i] = max(np.abs(u[i - 1],u[i]))
        # c[i] = max(abs(u[i - 1]), abs(u[i]))
    return c

# Calculate fluxes
def get_fluxes(u):
    f = 0.5 * np.power(u, 2)
    return f

# Riemann solver: Rusanov
def rusanov(fL,fR,u,uL,uR):
    f = np.zeros_like(u)

    c = get_wavespeed(u)

    # get interface fluxes (Rusanov) at i-1/2，故遍历时要遍历到最后一个物理网格的后一个网格
    f[3:-2] = 0.5*(fL[2:-3]+fR[3:-2])-0.5*c[3:-2]*(uR[3:-2]-uL[2:-3])
    return f



