# 开发人员：leo
# 开发时间：2023/4/18 14:05

import numpy as np
from weno5_scheme import weno5L, weno5R
from Riemann_solvers import *

# Calculate right hand side terms of the Euler equations

"""
LU = -udu/dx  但考虑了u[i]的速度方向，u[i]>0时用uL；u[i]<0时用uR
LU = -1/dx*(u^plus*(uLR-uLL)+u^minus*(uRR-uRL))
"""


def get_LU(dx, q, gamma):
    LU = np.zeros_like(q[:, :])

    # WENO Reconstruction
    qL = weno5L(q)  # construct left state at i-1/2
    qR = weno5R(q)  # construct right state at i-1/2

    # Computing fluxes
    fL = get_fluxes(qL, gamma)
    fR = get_fluxes(qR, gamma)

    # compute flux using Roe Riemann solver (flux at interface i-1/2)
    # f = roe(gamma, qL, qR, fL, fR)

    # compute flux using HLLC Riemann solver (flux at interface i-1/2) 精度比roe更高
    f = HLLC(gamma, qL, qR, fL, fR)

    # compute flux using Rusanov's Riemann Solver (Gudanov scheme) 精度比Roe和HLLC低但高效
    # f = gudanov(gamma, qL, qR, fL, fR)

    # compute LU using Riemann solver
    for k in range(3):
        LU[k, 3:-3] = -(f[k, 4:-2] - f[k, 3:-3]) / dx
    return LU

# Calculate fluxes
def get_fluxes(q, gamma):
    f = np.zeros_like(q)
    for i in range(3, len(q[0, :]) - 2):
        p = (gamma - 1.0) * (q[2, i] - 0.5 * q[1, i] * q[1, i] / q[0, i])
        f[0, i] = q[1, i]
        f[1, i] = q[1, i] * q[1, i] / q[0, i] + p
        f[2, i] = q[1, i] * q[2, i] / q[0, i] + p * q[1, i] / q[0, i]
    return f
