# 开发人员：leo
# 开发时间：2023/5/3 11:20

import numpy as np
from matplotlib import pyplot as plt


def plotStates(x, rho, u, p):
    states = (rho, u, p)
    states_name = (r"$\rho$", r"$u$", r"$p$")
    plt.figure(figsize=(15, 4), dpi=100)
    for i, state in enumerate(states):
        plt.subplot(1, 3, i + 1)
        plt.plot(x, state, linewidth=1)
        plt.xlabel("$x$")
        plt.ylabel(states_name[i])
        plt.xlim([0.0, 1.0])
    plt.show()

def plotStates2(x, rho, u, p, e):
    states = (rho, u, p, e)
    states_name = (r"$\rho$", r"$u$", r"$p$", r"$e$")
    plt.figure(figsize=(8, 6), dpi=100)
    for i, state in enumerate(states):
        plt.subplot(2, 2, i + 1)
        plt.plot(x, state, linewidth=1)
        plt.xlabel("$x$")
        plt.ylabel(states_name[i])
        plt.xlim([0.0, 1.0])
    plt.show()

def plotq(x,q):
    states = (q[0,:], q[1,:], q[2,:])
    states_name = (r"$\rho$", r"$\rho u$", r"$\rho e$")
    plt.figure(figsize=(15, 4), dpi=100)
    for i, state in enumerate(states):
        plt.subplot(1, 3, i + 1)
        plt.plot(x, state, linewidth=1)
        plt.xlabel("$x$")
        plt.ylabel(states_name[i])
        plt.xlim([0.0, 1.0])
    plt.show()