# 开发人员：leo
# 开发时间：2023/4/7 22:21

import numpy as np


def get_u_plux(u):
    return np.maximum(u,0)

def get_u_minus(u):
    return np.minimum(u,0)


a = [6,-5,7,-6,-9,2]
b = get_u_plux(a)
c = get_u_minus(a)


print(a)
print(b)
print(c)

def update_BC(u, BC_type):
    u[2], u[-3] = 0.0, 0.0
    if BC_type == "dirichlet_BC":
        # dirichlet boundary condition for ghost cells
        u[0] = 3 * u[2] - 2 * u[3]
        u[1] = 2 * u[2] - u[3]
        u[-1] = 3 * u[-3] - 2 * u[-4]
        u[-2] = 2 * u[-3] - u[-4]
    if BC_type == "periodic_BC":
        u[0] = u[-4]
        u[1] = u[-3]
        u[-1] = u[3]
        u[-2] = u[2]

ut = [1,2,3,4,5,6,7,8,9,10,11]
update_BC(ut,"dirichlet_BC")
print(ut)

cc = [1,2,3,4,5,6,7,8,9,10]
print(len(cc),cc[3], cc[-1], "定位")
for i in range(1, len(cc)-2):
    print(cc[i])