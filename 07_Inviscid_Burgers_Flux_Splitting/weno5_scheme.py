# 开发人员：leo
# 开发时间：2023/4/18 22:19

"""
调用方法：
uL = weno5L(u)
uR = weno5R(u)
输入数组u，返回uL和uR
"""

import numpy as np

eps = 1.0e-6


def weno5L(u):
    uL = np.zeros_like(u)
    for i in range(2,len(u)-3):  # uL从index=2开始更新，index=-4结束（包含2和-4）
        uL[i] = weightsL(u[i-2], u[i-1], u[i], u[i+1], u[i+2])
        # uL[i] = weights(u[i - 2], u[i - 1], u[i], u[i + 1], u[i + 2])  # 用于验证通用函数weights，貌似效果还行
    return uL



def weno5R(u):
    uR = np.zeros_like(u)
    for i in range(3, len(u) - 2):  # # uR从index=3开始更新，index=-3结束（包含3和-3）
        uR[i] = weightsR(u[i - 2], u[i - 1], u[i], u[i + 1], u[i + 2])
        # uR[i] = weights(u[i + 2], u[i + 1], u[i], u[i - 1], u[i - 2])  # 用于验证通用函数weights，貌似效果还行
    return uR



"""
# 其实根据对称性，weightsL和weightsR可以写成同一个函数
可参考https://github.com/fengyiqi/cfd_practice/blob/main/jupyter/05_Inviscid_Burgers_WENO/weno5_scheme_test.py
但本程序为了良好的阅读性写成了两个函数，且有一定的可扩展性，方便扩展不对称的权重函数
合成同一个函数的weights写在更下方，初步验证效果不错，但正确性需要进一步验证
"""


# v1,v2,v3,v4,v5: point values of v in the stencil (v3 = v[i])
# uL: left side numerical reconstruction at (i+1/2) interface     注意区别
def weightsL(v1, v2, v3, v4, v5):
    # smoothness indicators
    beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    beta3 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2

    # computing nonlinear weights w1,w2,w3
    a1 = 0.1 / (beta1 + eps) ** 2
    a2 = 0.6 / (beta2 + eps) ** 2
    a3 = 0.3 / (beta3 + eps) ** 2

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    # candidate stencils
    q1 = v1 / 3.0 - 7.0 / 6.0 * v2 + 11.0 / 6.0 * v3
    q2 = -v2 / 6.0 + 5.0 / 6.0 * v3 + v4 / 3.0
    q3 = v3 / 3.0 + 5.0 / 6.0 * v4 - v5 / 6.0

    # reconstructed value at interface at (i+1/2)
    uL = (w1 * q1 + w2 * q2 + w3 * q3)

    return uL


# v1,v2,v3,v4,v5: point values of v in the stencil (v3 = v[i])
# uL: left side numerical reconstruction at (i-1/2) interface     注意区别
def weightsR(v1, v2, v3, v4, v5):
    # smoothness indicators
    beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    beta3 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2

    # computing nonlinear weights w1,w2,w3
    a1 = 0.3 / (beta1 + eps) ** 2
    a2 = 0.6 / (beta2 + eps) ** 2
    a3 = 0.1 / (beta3 + eps) ** 2

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    # candiate stencils
    q1 = -v1 / 6.0 + 5.0 / 6.0 * v2 + v3 / 3.0
    q2 = v2 / 3.0 + 5.0 / 6.0 * v3 - v4 / 6.0
    q3 = 11.0 / 6.0 * v3 - 7.0 / 6.0 * v4 + v5 / 3.0

    # reconstructed value at interface at (i-1/2)
    uR = (w1 * q1 + w2 * q2 + w3 * q3)

    return uR


# 尝试写成同一个函数，对于uR需要逆序传值，正确性待验证
# v1,v2,v3,v4,v5: point values of v in the stencil (v3 = v[i])
def weights(v1, v2, v3, v4, v5):
    # smoothness indicators
    beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    beta3 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2

    # computing nonlinear weights w1,w2,w3
    a1 = 0.1 / (beta1 + eps) ** 2
    a2 = 0.6 / (beta2 + eps) ** 2
    a3 = 0.3 / (beta3 + eps) ** 2

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    # candidate stencils
    q1 = v1 / 3.0 - 7.0 / 6.0 * v2 + 11.0 / 6.0 * v3
    q2 = -v2 / 6.0 + 5.0 / 6.0 * v3 + v4 / 3.0
    q3 = v3 / 3.0 + 5.0 / 6.0 * v4 - v5 / 6.0

    # reconstructed value at interface
    f = (w1 * q1 + w2 * q2 + w3 * q3)

    return f
