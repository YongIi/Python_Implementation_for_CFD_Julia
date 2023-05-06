# 开发人员：leo
# 开发时间：2023/5/5 21:45

import numpy as np
from renew_states import renew_states

def HLLC(gamma, qL, qR, fL, fR):  # All variable values are located at i-1/2
    f = np.zeros_like(qL)

    Ds = np.zeros(3)  # Dstar
    Ds[0], Ds[1] = 0.0, 1.0

    # Left and right states at i-1/2
    rhLL, uuLL, eeLL, ppLL, hhLL, ccLL = renew_states(qL, gamma)  # 部分变量在本求解器中并未使用
    rhRR, uuRR, eeRR, ppRR, hhRR, ccRR = renew_states(qR, gamma)

    for i in range(3, len(qL[0, :]) - 2):  # uR从index=3开始更新，index=-3结束（包含3和-3）
        # lower and upper bounds on the characteristics speeds
        # compute SL and SR
        SL = min(uuLL[i], uuRR[i]) - max(ccLL[i], ccRR[i])
        SR = max(uuLL[i], uuRR[i]) + max(ccLL[i], ccRR[i])

        # middle wave of speed Sstar
        # compute compound speed
        SP = (ppRR[i] - ppLL[i] + rhLL[i] * uuLL[i] * (SL - uuLL[i]) - rhRR[i] * uuRR[i] * (SR - uuRR[i]))\
             /(rhLL[i] * (SL - uuLL[i]) - rhRR[i] * (SR - uuRR[i]))  # never get zero

        # compute compound pressure (mean pressure)
        PLR = 0.5 * (ppLL[i] + ppRR[i] + rhLL[i] * (SL - uuLL[i]) * (SP - uuLL[i])\
                     + rhRR[i] * (SR - uuRR[i]) * (SP - uuRR[i]))

        # compute D
        Ds[2] = SP

        # get interface fluxes at i-1/2，故遍历时要遍历到最后一个物理网格的后一个网格
        if SL >= 0.0:
            for k in range(3):
                f[k,i] = fL[k,i]
        elif SR <= 0.0:
            for k in range(3):
                f[k,i] = fR[k,i]
        elif ((SP>=0.0)and(SL<=0.0)):
            for k in range(3):
                f[k,i] = (SP*(SL*qL[k,i]-fL[k,i])+SL*PLR*Ds[k])/(SL-SP)
        elif ((SP<=0.0)and(SR>=0.0)):
            for k in range(3):
                f[k, i] = (SP*(SR*qR[k,i]-fR[k,i])+SR*PLR*Ds[k])/(SR-SP)

    return f