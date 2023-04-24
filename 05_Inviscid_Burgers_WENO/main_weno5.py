# 开发人员：leo
# 开发时间：2023/4/17 21:47

# inviscid Burgers equation

"""
du/dt+u*du/dx=0

The hyperbolic equations admit discontinuities, and the
numerical schemes used for solving hyperbolic PDEs need to be higher-order accurate for smooth
solutions, and non-oscillatory for discontinuous solutions

"""

import numpy as np
from rk3 import rk3
import matplotlib.pyplot as plt

# from weno5_scheme import weno5_reconstruction
# from rk3 import rk3

# 要用FDM的思想去理解index，计算域共nx+1+2*ghost_cells_n个点，且首尾用作ghost cells

ghost_cells_n = 2  # 理应采用3个ghost cells，但是第一个物理网格是固定值边界

x_l, x_r, t = 0, 1, 0.25
nx = 200
dt = 0.0001
dx = (x_r - x_l) / nx
nt = int(t / dt)

# index for x
# [-2, -1, 0, 1,... ,N-1,N, N+1, N+2]
x = np.linspace(x_l - ghost_cells_n * dx, x_r + ghost_cells_n * dx, nx + 1 + 2 * ghost_cells_n)
u = np.zeros((nt+1, nx + 1 + 2 * ghost_cells_n))
# print(len(x),len(u[0]))

# IC
u[0, :] = np.sin(2*np.pi*x)
# BC
u[0, 2], u[0, -3] = 0.0, 0.0
# dirichlet boundary condition for ghost cells
u[0, 1] = 2*u[0,2]-u[0,3]
u[0, 0] = 3*u[0,2]-2*u[0,3]
u[0, -1] = 3*u[0,-3]-2*u[0,-4]
u[0, -2] = 2*u[0,-3]-u[0,-4]

rk3(u,nt,dt,dx,"dirichlet_BC")  # 不需要等号赋值
# rk3(u,nt,dt,dx,"periodic_BC")  # 采用周期性边界条件

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