import numpy as np
import pandas as pd
from gab_road_emission_pricing import genetic_algorithm
from best_response_decomposition import best_response_decomposition

# 读取交通网络拓扑结构
node_link_excel = pd.read_excel(
    'link_node_8.xlsx', sheet_name='Sheet1', index_col=0, header=None)
i_rs = pd.read_excel('link_node_8.xlsx',
                     sheet_name='Sheet2', index_col=0, header=None)
node_link = node_link_excel.iloc[0:4]
link_type = node_link_excel.iloc[[5]]
road_type = node_link_excel.iloc[[4]]

L = node_link.values
# I_rs指表示OD对的指示向量
I_rs = i_rs.values
link_type_v = link_type.values.reshape((-1,))
n_l = 12
d = np.zeros(n_l)
bypass = np.zeros(n_l)
for i in range(n_l):
    if link_type_v[i] == 0:
        d[i] = 1
    if link_type_v[i] == 2:
        bypass[i] = 1

energy_demand = 50 * 0.001
# 假设充电功率是350kW, 计算充电时间
tc = energy_demand * 1000 / 350 * 60
# 给定路段t0和c
t_0 = np.array([11, 11+tc, 9, 9+tc, 12, 12+tc, 15, 15+tc, 8, 8+tc, 9, 9+tc])
c = np.array([35, 20, 40, 20, 45, 20, 40, 20, 15, 20, 15, 20])

# OD出行需求
q_gaso = np.array([30, 25])
q_elec = np.array([15, 10])

# 电网相关参数
num_bus = 6
num_line = 6
a = 0.3
b = 150
rho = 140
# 矩阵中第i行表示节点i有哪些子节点，第j列表示节点j的父节点是谁
child_bus = np.array(
    (
        (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 1, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0)
    )
)
# 表示第一个节点是连接到主网的节点
pi = np.array([1, 0, 0, 0, 0, 0])
r = np.array([0.081, 0.066, 0.102, 0.061, 0.058, 0.093])
x = np.array([0.101, 0.092, 0.143, 0.091, 0.077, 0.124])
p_dt = 0.1 * 100
q_d = 0.1 * 100
U_0 = 1.04 ** 2 * 100
p_gn = 0.
p_gm = 1.5 * 100
q_gn = -1. * 100
q_gm = 1. * 100
U_n = 0.88 ** 2 * 100
U_m = 1.05 ** 2 * 100
S = 1.5 * 100


# decimal: 道路收费需要精确到小数点后几位
# population_size: 种群中染色体数量
# GA_MAX_ITER: 遗传算法最大迭代次数
# BRD_MAX_ITER: 最佳响应分解算法最大迭代次数
best_toll, best_fitness, fitness_iter = genetic_algorithm(num_link=12, x_min=0, x_max=10, decimal=3, population_size=30, GA_MAX_ITER=100,
                                                          epsilon=0.5, BRD_MAX_ITER=20, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                          p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
                                                          num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v, t_0=t_0, capacity=c, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)

ue, ue_gaso, lmp, D, f, p_g, P, q_g, Q, U = best_response_decomposition(epsilon=0.5, MAX_ITER=20, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                                 p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=best_toll,
                                                                 num_link=12, num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v, t_0=t_0, capacity=c, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec,
                                                                 verbose=False)

print('ue:')
print(ue)
print('lmp:')
print(lmp)
print('toll:')
print(best_toll)

np.savetxt('results/onetime/ue.txt', ue)
np.savetxt('results/onetime/lmp.txt', lmp)
np.savetxt('results/onetime/toll.txt', best_toll)
np.savetxt('results/onetime/fit.txt', fitness_iter)
