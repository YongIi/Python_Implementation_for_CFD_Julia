# 开发人员：leo
# 开发时间：2023/4/18 21:14

def update_BC(u, BC_type):
    if BC_type == "dirichlet_BC": # 该BC不对所有求解器通用
        # dirichlet boundary condition for ghost cells
        u[0] = 3 * u[2] - 2 * u[3]
        u[1] = 2 * u[2] - u[3]
        u[-1] = 3 * u[-3] - 2 * u[-4]
        u[-2] = 2 * u[-3] - u[-4]
    if BC_type == "periodic_BC":
        u[0] = u[-6]
        u[1] = u[-5]
        u[2] = u[-4]
        u[-1] = u[5]
        u[-2] = u[4]
        u[-3] = u[3]