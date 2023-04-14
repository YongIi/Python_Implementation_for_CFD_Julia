# 开发人员：leo
# 开发时间：2023/4/7 16:32

import numpy as np
import matplotlib.pyplot as plt

x_l, x_r, t = -1, 1, 1
dx, dt = 0.025, 0.0025
nx = int((x_r - x_l) / dx)
nt = int(t / dt)
print('nx={0}, nt={1}'.format(nx, nt))

alpha = 1 / np.pi ** 2
beta = alpha * dt / dx ** 2

u = np.zeros((nt + 1, nx + 1))  # 2D array
x = np.linspace(x_l, x_r, nx + 1)

# IC
u[0, :] = -np.sin(np.pi * x)
u[0, 0], u[0, -1] = 0.0, 0.0

# FTCS
for i in range(1, nt + 1):
    for j in range(1, nx):  # 第一个点和最后一个点是ghost cells
        u[i, j] = u[i - 1, j] + beta * (u[i - 1, j + 1] - 2 * u[i - 1, j] + u[i - 1, j - 1])
    # BC at x=-1 and x=1
    u[i, 0], u[i, -1] = 0.0, 0.0

# exact solution
u_exact = -np.exp(-t) * np.sin(np.pi * x)
# Discretization error
error = np.abs(u_exact - u[-1, :])

# postprocessing
plt.figure(figsize=(10, 4), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(x, u_exact, "k-", linewidth=1.0, label="Exact")
plt.scatter(x, u[-1, :], facecolor="none", edgecolor="blue", s=20, linewidths=0.5, label="FTCS")
plt.xlabel("$x$")
plt.ylabel("$u$")
plt.title("Solution field")
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.scatter(x, error, facecolor="none", edgecolor="red", s=20, linewidths=0.5)
plt.ylabel(r"$\epsilon$")
plt.xlabel("$x$")
plt.title("Discretization error")
plt.tight_layout()
plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

plt.show()