# 开发人员：leo
# 开发时间：2023/4/7 21:35

import numpy as np
import matplotlib.pyplot as plt

# 要用FDM的思想去理解index，计算域共nx+1个点，且首尾用作ghost cells

def rk3(nx, nt, dx, dt, u, alpha):
    #un, ut, LU = np.zeros(nx + 1), np.zeros(nx + 1), np.zeros(nx + 1)
    un = u[0, :].copy()
    k = 0  # record index
    # dirichlet boundary condition for temporary array
    #ut[0] = 0.0
    #ut[-1] = 0.0

    for i in range(1, nt + 1):
        # 1st step
        LU = get_LU(dx, un, alpha)
        ut = un + dt * LU
        # 2nd step
        LU = get_LU(dx, ut, alpha)
        ut = 0.75 * un + 0.25 * ut + 0.25 * dt * LU
        # 3rd step
        LU = get_LU(dx, ut, alpha)
        un = (1.0 / 3.0) * un + (2.0 / 3.0) * ut + (2.0 / 3.0) * dt * LU


        k+=1
        u[k, :] = un[:]

    return u



def get_LU(dx, u, alpha):
    LU = np.zeros_like(u)
    LU[1:-1] = alpha * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return LU


x_l, x_r, t = -1, 1, 1
dx, dt = 0.025, 0.0025
nx = int((x_r - x_l) / dx)
nt = int(t / dt)
print(nx)

alpha = 1 / np.pi ** 2

u = np.zeros((nt + 1, nx + 1))  # index=0, -1时为单个的ghost cell
x = np.linspace(x_l, x_r, nx + 1)
print(len(x),len(u[0]))


# IC
u[0, :] = -np.sin(np.pi * x, dtype=np.double)
u[0, 0], u[0, -1] = 0.0, 0.0

# exact solution and error
u_exact = -np.exp(-t) * np.sin(np.pi * x)
# error = np.abs(u_exact - u[-1, :])
error = np.zeros_like(x)

u = rk3(nx, nt, dx, dt, u, alpha)

error = np.abs(u_exact - u[-1, :])

plt.figure(figsize=(10, 4), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(x, u_exact, "k-", linewidth=1.0, label="Solution")
plt.scatter(x, u[-1, :], facecolor="none", edgecolor="blue", s=20, linewidths=0.5, label="RK3")
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
