# 开发人员：leo
# 开发时间：2023/5/3 23:06

# Riemann solver: Roe's approximate Riemann solver

import numpy as np
from renew_states import renew_states
from math import sqrt

def roe(gamma, qL, qR, fL, fR):  # All variable values are located at i-1/2
    f = np.zeros_like(qL)
    gm = gamma - 1.0
    # Left and right states at i-1/2
    rhLL, uuLL, eeLL, ppLL, hhLL = renew_states(qL, gamma)
    rhRR, uuRR, eeRR, ppRR, hhRR = renew_states(qR, gamma)

    for i in range(3, len(qL[0,:]) - 2):  # uR从index=3开始更新，index=-3结束（包含3和-3）
        V, dd, dF = np.zeros(3), np.zeros(3), np.zeros(3)

        # 分母
        denominator = 1.0 / (sqrt(abs(rhLL[i])) + sqrt(abs(rhRR[i])))

        # Roe averaged values
        uu = (sqrt(abs(rhLL[i])) * uuLL[i] + sqrt(abs(rhRR[i])) * uuRR[i]) * denominator  # u: velocity
        hh = (sqrt(abs(rhLL[i])) * hhLL[i] + sqrt(abs(rhRR[i])) * hhRR[i]) * denominator  # h: enthalpy
        cc = sqrt(abs(gm * (hh - 0.5 * uu * uu)))  # c: sound speed

        # absolute values of eigenvalues
        D11 = abs(uu)
        D22 = abs(uu + cc)
        D33 = abs(uu - cc)

        #  intermediate variable
        beta = 0.5 / (cc * cc)
        phi2 = 0.5 * gm * uu * uu

        # Right eigenvector matrix
        R11, R21, R31 = 1.0, uu, phi2/gm
        R12, R22, R32 = beta, beta*(uu+cc), beta*(hh+uu*cc)
        R13, R23, R33 = beta, beta*(uu-cc), beta*(hh-uu*cc)

        # Left eigenvector matrix
        L11, L12, L13 = 1.0-phi2/(cc*cc), gm*uu/(cc*cc), -gm/(cc*cc)
        L21, L22, L23 = phi2-uu*cc, cc-gm*uu, gm
        L31, L32, L33 = phi2+uu*cc, -cc-gm*uu, gm

        # 0.5*(qR-qL) in Equation (68)
        for k in range(3):
            V[k] = 0.5*(qR[k,i]-qL[k,i])

        # 矩阵相乘，方向从右往左乘
        dd[0] = D11*(L11*V[0]+L12*V[1]+L13*V[2])
        dd[1] = D22*(L21*V[0]+L22*V[1]+L23*V[2])
        dd[2] = D33*(L31*V[0]+L32*V[1]+L33*V[2])

        dF[0] = R11*dd[0]+R12*dd[1]+R13*dd[2]
        dF[1] = R21*dd[0]+R22*dd[1]+R23*dd[2]
        dF[2] = R31*dd[0]+R32*dd[1]+R33*dd[2]

        # get interface fluxes at i-1/2，故遍历时要遍历到最后一个物理网格的后一个网格
        for k in range(3):
            f[k,i] = 0.5*(fR[k,i]+fL[k,i])-dF[k]
    return f

"""
# Riemann solver: Rusanov
def rusanov(fL, fR, u, uL, uR):
    f = np.zeros_like(u)

    c = get_wavespeed(u)

    # get interface fluxes (Rusanov) at i-1/2，故遍历时要遍历到最后一个物理网格的后一个网格
    f[3:-2] = 0.5 * (fL[2:-3] + fR[3:-2]) - 0.5 * c[3:-2] * (uR[3:-2] - uL[2:-3])
    return f

# propagation speed
def get_wavespeed(u):
    c = np.zeros_like(u)
    for i in range(2, len(u) - 2):  # 从index=2开始，index=-3结束（包含2和-3）
        c[i] = max(abs(u[i - 2]), abs(u[i - 1]), abs(u[i]), abs(u[i + 1]), abs(u[i + 2]))
        # c[i] = max(abs(u[i - 1],u[i]))
        # c[i] = max(abs(u[i - 1]), abs(u[i]))
    return c
    
"""
