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

cc2=0.5*np.power(cc,2)
print(cc2)

print("--------")

def get_wavespeed(u):
    alpha = np.zeros_like(u)
    for i in range(3,len(u)-3):  # 从index=3开始，index=-4结束（包含2和-4）
        alpha[i] = max(abs(u[i-2]),abs(u[i-1]),abs(u[i]),abs(u[i+1]),abs(u[i+2]))
    return alpha

u = [3,-4,5,-6,7,-4,3,4,5,-6,7,8,9,4,-3,3,-5,6]

aaa = get_wavespeed(u)
print(aaa)

for i in range(3):
    print(i)

oo = [7,5,44,2,8,5,4,5,7,4,6,9,5,7,6,5,7,4]
for i in range(0,3):
    print(oo[i])
print("---")
def BC(hh):
    for i in range(len(hh) - 3, len(hh)):
        hh[i] = 0
        print(hh[i])

BC(oo)
print(oo)

c1 = [0,1,2,3,4]
print(len(c1))

for i in range(3):
    print(i)
