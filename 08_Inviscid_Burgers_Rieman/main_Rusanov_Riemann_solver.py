# 开发人员：leo
# 开发时间：2023/4/24 21:11

# Inviscid Burgers Equation: Conservative Form

# rusanov riemann solver using rk3 and weno5

"""
du/dt+df/dx=0  where f=u^2/2

using FVM

"""

import numpy as np
import matplotlib.pyplot as plt
from rk3 import rk3

ghost_cells_n = 3

x_min, x_max, T = 0, 1, 0.25
nx = 150
dt = 0.0001
dx = (x_max - x_min) / nx
nt = int(T / dt)

x = np.linspace(dx / 2 - ghost_cells_n * dx, x_max - dx / 2 + ghost_cells_n * dx, nx + 2 * ghost_cells_n)
u = np.zeros((nt + 1, nx + 2 * ghost_cells_n))
# print("包含ghost的总网格数：",len(x),len(u[0]))

# IC
u[0, :] = np.sin(2 * np.pi * x)
# BC periodic bc
u[0, 0] = u[0, -6]
u[0, 1] = u[0, -5]
u[0, 2] = u[0, -4]
u[0, -1] = u[0, 5]
u[0, -2] = u[0, 4]
u[0, -3] = u[0, 3]

rk3(u,nt,dt,dx,"periodic_BC")

# postProcessing
ts = np.linspace(0.025, 0.25, 10)
plt.figure(figsize=(6, 4), dpi=100)
for i, t_stamp in enumerate(ts):  # 用enumerate函数来循环ts数组中的元素，并获取每个元素的下标和值
    index = int(t_stamp / dt)
    plt.plot(x, u[index], linewidth=0.8, label="t="+format(t_stamp, ".4f"))
plt.legend(fontsize=7)
plt.xlim([0.0, 1.0])
plt.ylabel("$x$")
plt.ylabel("$u$")
plt.tight_layout()
plt.show()

