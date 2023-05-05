# 开发人员：leo
# 开发时间：2023/5/3

# Roe riemann solver using rk3 and weno5

import numpy as np
import myplot
from renew_states import renew_states
from rk3 import rk3

ghost_cells_n = 3

x_min, x_max, T = 0, 1, 0.20
nx = 256
dt = 0.0001
dx = (x_max - x_min) / nx
nt = int(T / dt)

ns = 20  # 计算结果保存的个数
ds = T / ns  # 输出时间间隔

# Sod's Riemann problem
gamma = 1.4  # specific gas ratio
# Left side
rhoL = 1.0
uL = 0.0
pL = 1.0
# Right side
rhoR = 0.125
uR = 0.0
pR = 0.1

x = np.linspace(dx / 2 - ghost_cells_n * dx, x_max - dx / 2 + ghost_cells_n * dx, nx + 2 * ghost_cells_n)
# print("包含ghost的总网格数：",len(x),len(u[0]))
rho = np.zeros(nx + 2 * ghost_cells_n)
u = np.zeros(nx + 2 * ghost_cells_n)
p = np.zeros(nx + 2 * ghost_cells_n)
e = np.zeros(nx + 2 * ghost_cells_n)
q = np.zeros((ns + 1, 3, nx + 2 * ghost_cells_n))  # q[时间, 守恒量, 空间坐标]
print("q的维度是：", q.shape)

# IC
"""
# 方法一：切片法。切片法效率高，但稍微复杂点
# 数组切片后没有生成新的数组，因为切片是右开区间，所以要加上1， # index = x/dx
rho[:int(0.5 / dx + 1)+ghost_cells_n] = rhoL
rho[int(0.5 / dx)+ghost_cells_n:] = rhoR
u[:int(0.5 / dx + 1)+ghost_cells_n] = uL
u[int(0.5 / dx)+ghost_cells_n:] = uR
p[:int(0.5 / dx + 1)+ghost_cells_n] = pL
p[int(0.5 / dx)+ghost_cells_n:] = pR
e = p/(rho*(gamma-1.0)) + 0.5*u*u
#conservative variables
q[0,0,:] = rho
q[0,1,:] = rho*u
q[0,2,:] = rho*e
print(len(x))
print(int(0.5 / dx + 1)+ghost_cells_n)
"""
# 方法二：循环法。循环法看似代码量小，但是会效率比切片法低
for i, pos in enumerate(x):
    rho[i] = rhoL if pos < 0.5 else rhoR
    u[i] = uL if pos < 0.5 else uR
    p[i] = pL if pos < 0.5 else pR
    e[i] = p[i] / (rho[i] * (gamma - 1.0)) + 0.5 * u[i] * u[i]
    # conservative variables
    q[0, 0, i] = rho[i]
    q[0, 1, i] = rho[i] * u[i]
    q[0, 2, i] = rho[i] * e[i]

rk3(gamma,q,ns,nt,dt,dx,"symmetric_BC")

# postProcessing
# IC
# myplot.plotStates(x, rho, u, p)
myplot.plotStates2(x, rho, u, p, e)
# myplot.plotq(x,q[0,:,:])

# results
rho1, u1, e1, p1, h1 = renew_states(q[-1,:,:], gamma)
myplot.plotStates2(x, rho1, u1, p1, e1)
