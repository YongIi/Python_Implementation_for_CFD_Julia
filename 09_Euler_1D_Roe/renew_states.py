# 开发人员：leo
# 开发时间：2023/5/4 11:20
import numpy as np


def renew_states(q, gamma):
    rho, u, e, p, h = np.zeros_like(q[0,:]),np.zeros_like(q[0,:]),np.zeros_like(q[0,:]),np.zeros_like(q[0,:]),np.zeros_like(q[0,:])
    gm = gamma - 1.0

    for i in range(3, len(q[0,:]) - 2):
        rho[i] = q[0, i]
        u[i] = q[1, i] / q[0, i]
        e[i] = q[2, i] / q[0, i]
        p[i] = gm * (rho[i] * e[i] - 0.5 * rho[i] * u[i] * u[i])
        h[i] = e[i] + p[i] / rho[i]

    return rho, u, e, p, h

