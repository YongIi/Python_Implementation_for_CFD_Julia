# 开发人员：leo
# 开发时间：2023/4/18 21:14

def update_BC(u, BC_type):
    nx = len(u[0,:]) - 6
    if BC_type == "dirichlet_BC": # 该BC不对所有求解器通用
        # dirichlet boundary condition for ghost cells
        u[0] = 3 * u[2] - 2 * u[3]
        u[1] = 2 * u[2] - u[3]
        u[-1] = 3 * u[-3] - 2 * u[-4]
        u[-2] = 2 * u[-3] - u[-4]
    if BC_type == "periodic_BC": # 该BC不对所有求解器通用
        u[0] = u[-6]
        u[1] = u[-5]
        u[2] = u[-4]
        u[-1] = u[5]
        u[-2] = u[4]
        u[-3] = u[3]
    if BC_type == "symmetric_BC":
        # for left side
        for i in range(3):
            for k in range(3): # 标量相等
                u[k, i] = u[k, 5 - i]
            u[1, i] = -u[1, 5 - i]  # 矢量反号
        # for right side
        for i in range(len(u[0,:]) - 3, len(u[0,:])):
            for k in range(3): # 标量相等
                u[k,i] = u[k,2*nx+5-i]
            u[1, i] = -u[1,2*nx+5-i] # 矢量反号

        """
        # 方法二：切片法
        u[:,0] = u[:,5]
        u[:,1] = u[:,4]
        u[:,2] = u[:,3]
        u[:,-3] = u[:,-4]
        u[:, -2] = u[:, -5]
        u[:, -1] = u[:, -6]
        """

