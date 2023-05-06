# 开发人员：leo
# 开发时间：2023/5/6 9:52

# Rusanov's Riemann Solver  植入简单计算高效，但精度不如HLLC和reo
"""
f = 0.5*(fL+fR)-0.5*c*(uR-uL)
"""

import numpy as np
from renew_states import renew_states
from math import sqrt

def gudanov(gamma, qL, qR, fL, fR):
    f = np.zeros_like(qL)

    # compute wave speed
    c = get_wavespeed(gamma, qL, qR)

    # get interface fluxes (Rusanov) at i-1/2，故遍历时要遍历到最后一个物理网格的后一个网格
    for k in range(3):
        f[k,3:-2] = 0.5*(fL[k,3:-2]+fR[k,3:-2])-0.5*c[3:-2]*(qR[k,3:-2]-qL[k,3:-2])

    return f

# propagation speed
def get_wavespeed(gamma, qL, qR):
    c = np.zeros_like(qL[0,:])
    gm = gamma-1.0

    # Left and right states at i-1/2
    rhLL, uuLL, eeLL, ppLL, hhLL, ccLL = renew_states(qL, gamma)  # 部分变量在本求解器中并未使用
    rhRR, uuRR, eeRR, ppRR, hhRR, ccRR = renew_states(qR, gamma)

    for i in range(3, len(qL[0, :]) - 2):  # uR从index=3开始更新，index=-3结束（包含3和-3）

        # 分母
        denominator = 1.0 / (sqrt(abs(rhLL[i])) + sqrt(abs(rhRR[i])))

        # Roe averaged values
        uu = (sqrt(abs(rhLL[i])) * uuLL[i] + sqrt(abs(rhRR[i])) * uuRR[i]) * denominator  # u: velocity
        hh = (sqrt(abs(rhLL[i])) * hhLL[i] + sqrt(abs(rhRR[i])) * hhRR[i]) * denominator  # h: enthalpy
        cc = sqrt(abs(gm * (hh - 0.5 * uu * uu)))  # c: sound speed

        # c[i] = abs(uu+cc)
        c[i] = max(abs(uu+cc),abs(uu-cc))

    return c

