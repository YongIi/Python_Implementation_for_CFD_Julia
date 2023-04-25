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
    f = np.zeros_like(u)
    fP = np.zeros_like(u)  # positive part of flux at the cell center
    fN = np.zeros_like(u)  # negative part of flux at the cell center

    # flux computed at cell center and positive and negative splitting
    f = 0.5*np.power(u,2)

    # wave speed at cell center
    alpha = get_wavespeed(u)

    # split the flux at cell center into positive and negative components 之所以这样分解也是因为f=0.5*u^2
    fP[3:-3] = 0.5*(f[3:-3]+alpha[3:-3]*u[3:-3])
    fN[3:-3] = 0.5*(f[3:-3]-alpha[3:-3]*u[3:-3])

    # WENO Reconstruction
    # compute upwind reconstruction for positive flux (left to right)
    fL = weno5L(fP)  # left side flux at the interface
    # compute downwind reconstruction for negative flux (right to left)
    fR = weno5R(fN)  # right side flux at the~interface

    # compute LU using flux splitting
    LU[3:-3] = -((fL[3:-3]-fL[2:-4])+(fR[4:-2]-fR[3:-3]))/dx
    return LU

def get_wavespeed(u):
    alpha = np.zeros_like(u)
    for i in range(3,len(u)-3):  # 从index=3开始，index=-4结束（包含2和-4）
        alpha[i] = max(abs(u[i-2]),abs(u[i-1]),abs(u[i]),abs(u[i+1]),abs(u[i+2]))
    return alpha

