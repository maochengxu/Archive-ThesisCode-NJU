import numpy as np
from tqdm import tqdm
import pandas as pd
from adaptive_path_generation import adaptive_path_generation
from optimal_power_flow_solver import opf_solver
import matplotlib.pyplot as plt


# 输出：UE，LMP，Toll
# 评价：系统总时间，系统总排放，电力系统总成本

# 交通系统
node_link_excel = pd.read_excel('link_node_8.xlsx', sheet_name='Sheet1', index_col=0, header=None)
i_rs = pd.read_excel('link_node_8.xlsx', sheet_name='Sheet2', index_col=0, header=None)
node_link = node_link_excel.iloc[0:4]
link_type = node_link_excel.iloc[[5]]
road_type = node_link_excel.iloc[[4]]

L = node_link.values
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
tc = energy_demand * 1000 / 350 * 60
t_0 = np.array([11, 11+tc, 9, 9+tc, 12, 12+tc, 15, 15+tc, 8, 8+tc, 9, 9+tc])
c = np.array([35, 20, 40, 20, 45, 20, 40, 20, 15, 20, 15, 20])

lmp_value = np.array([150, 150, 150, 150, 150, 150])
lmp_init = np.zeros(n_l)
j_ = 0
for i in range(n_l):
    if link_type_v[i] == 0:
        lmp_init[i] = lmp_value[j_]
        j_ = j_ + 1
# q_gaso = np.array([30, 25])
# q_elec = np.array([15, 10])

# 电力系统
num_bus = 6
num_line = 6
a = 0.3
b = 150
rho = 140
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


def load_injection(ue, charging_flag, energy_demand):
    p_dc = np.array([])
    for i in range(len(ue)):
        if charging_flag[i] == 1:
            p_dc = np.append(p_dc, ue[i] * energy_demand * 100)
    return p_dc


# 贪婪策略下的下层模型
def greedy_strategy(ue_prev, epsilon, MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
                    p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S, emission_toll,
                    num_link: int, num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
                    capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray, verbose=False):
    p_dc = load_injection(ue_prev, charging_flag, energy_demand)
    lmp, p_g, P, q_g, Q, U, f = opf_solver(num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x,
                          p_dt=p_dt, p_dc=p_dc, q_d=q_d, U_0=U_0, p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S)
    lmp_all = np.zeros(num_link)
    j_ = 0
    for i in range(num_link):
        if link_type[i] == 0:
            lmp_all[i] = lmp[j_]
            j_ = j_ + 1
    ue_new, delta_gaso, delta_elec, ue_gaso, ue_elec = adaptive_path_generation(num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type,
                                                                                t_0=t_0, capacity=capacity, lmp=lmp_all, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec, tau=emission_toll,
                                                                                verbose=verbose)
    return ue_new, ue_gaso, f, lmp

# 贪婪策略下的上层模型
from gab_road_emission_pricing import chromo_length, decoding_chromo, fitness_func, get_accu_prob_array, random_init, reproduction


def crossover(ue_prev, prob_cross, prob_muta, num_link, fitness_array, population: np.ndarray, x_min, x_max, decimal, chromo_len,
              epsilon, BRD_MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
              p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S,
              num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
              capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray):
    
    dice_cross = np.random.uniform(0, 1)
    if dice_cross <= prob_cross:
        for _ in range(10):
            # 选择
            accu_prob_array = get_accu_prob_array(fitness_array)
            father_index = reproduction(accu_prob_array, fitness_array)
            mother_index = reproduction(accu_prob_array, fitness_array)
            father = population[father_index]
            mother = population[mother_index]

            # 交叉
            # position = int(population.shape[1] * np.random.uniform(0, 1))
            position = 14 * int(num_link * np.random.uniform(0, 1))
            father_part_1 = father[:position]
            father_part_2 = father[position:]
            mother_part_1 = mother[:position]
            mother_part_2 = mother[position:]
            child_1 = np.concatenate((father_part_1, mother_part_2))
            child_2 = np.concatenate((mother_part_1, father_part_2))

            # 变异
            dice_muta = np.random.uniform(0, 1)
            if dice_muta <= prob_muta:
                muta_position = int(population.shape[1] * np.random.uniform(0, 1))
                child_1[muta_position] = not child_1[muta_position]
            dice_muta = np.random.uniform(0, 1)
            if dice_muta <= prob_muta:
                muta_position = int(population.shape[1] * np.random.uniform(0, 1))
                child_2[muta_position] = not child_2[muta_position]
            
            # 计算子代适应度
            toll_child_1 = np.zeros(num_link)
            for j in range(num_link):
                x_tmp = decoding_chromo(child_1[(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
                toll_child_1[j] = x_tmp
            toll_child_2 = np.zeros(num_link)
            for j in range(num_link):
                x_tmp = decoding_chromo(child_2[(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
                toll_child_2[j] = x_tmp

            ue_1, ue_gaso_1, f_1, lmp_1 = greedy_strategy(ue_prev=ue_prev, epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                            p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll_child_1,
                                                            num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
            fitness_child_1 = fitness_func(ue_gaso_1, ue_1, t_0, capacity, link_type, J)
            ue_2, ue_gaso_2, f_2, lmp_2 = greedy_strategy(ue_prev=ue_prev, epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                            p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll_child_2,
                                                            num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
            fitness_child_2 = fitness_func(ue_gaso_2, ue_2, t_0, capacity, link_type, J)

            # 找到适应度最小的两个
            min_arr = fitness_array.argsort()[0:2]

            if fitness_child_1 > fitness_child_2:
                tmp_fit = fitness_child_2
                fitness_child_2 = fitness_child_1
                fitness_child_1 = tmp_fit
                tmp_child = child_2
                child_2 = child_1
                child_1 = tmp_child

            if fitness_child_2 < fitness_array[min_arr[0]]:
                continue
            elif fitness_child_1 < fitness_array[min_arr[0]] and fitness_child_2 >= fitness_array[min_arr[0]]:
                population[min_arr[0]] = child_2
                fitness_array[min_arr[0]] = fitness_child_2
            elif fitness_child_1 >= fitness_array[min_arr[0]] and fitness_child_1 < fitness_array[min_arr[1]]:
                population[min_arr[0]] = child_2
                fitness_array[min_arr[0]] = fitness_child_2
            else:
                population[min_arr[0]] = child_1
                fitness_array[min_arr[0]] = fitness_child_1
                population[min_arr[1]] = child_2
                fitness_array[min_arr[1]] = fitness_child_2
    return population, fitness_array


def greedy_genetic_algorithm(ue_prev, num_link, x_min, x_max, decimal, population_size, GA_MAX_ITER,
                      epsilon, BRD_MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
                      p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S,
                      num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
                      capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray):
    # Initialization
    chromo_len = chromo_length(x_min, x_max, decimal)
    population = random_init(num_link, x_min, x_max,
                             population_size, chromo_len)
    fitness_iter = []

    # Initial fitness array
    fitness_array = np.array([])
    for i in tqdm(range(len(population))):
        toll = np.zeros(num_link)
        for j in range(num_link):
            x_tmp = decoding_chromo(
                population[i][(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
            toll[j] = x_tmp
        ue, ue_gaso, f, lmp = greedy_strategy(ue_prev=ue_prev, epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                      p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll,
                                      num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
        fitness = fitness_func(ue_gaso, ue, t_0, capacity, link_type, J)
        fitness_array = np.append(fitness_array, fitness)

    for i in tqdm(range(GA_MAX_ITER)):
        population, fitness_array = crossover(
            ue_prev, 0.95, 0.033, num_link, fitness_array, population, x_min, x_max, decimal, chromo_len,
            epsilon=epsilon, BRD_MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
            p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
            num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
        
        max_fitness = np.max(fitness_array)
        fitness_iter.append(max_fitness)
        print('Iteration ' + str(i) + ': ' + str(max_fitness))

    max_index = np.argmax(fitness_array)
    best_fitness = fitness_array[max_index]
    best_toll = np.zeros(num_link)
    for j in range(num_link):
        x_tmp = decoding_chromo(population[max_index][(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
        best_toll[j] = x_tmp
    ue, ue_gaso, f, lmp = greedy_strategy(ue_prev=ue_prev, epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=best_toll,
                                num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
    return best_toll, ue, ue_gaso, f, lmp


# 评价指标计算
def get_latency_with_all_types(x: np.ndarray, link_type: np.ndarray, t_0: np.ndarray, capacity: np.ndarray, J: float) -> np.ndarray:
    def latency_func_charging(x, t_0, c, J):
        if x < c:
            return t_0 * (1. + J * (x / (c - x)))
        else:
            return 1e8

    def latency_func_regular(x, t_0, c):
        return t_0 * (1. + 0.15 * (x / c) ** 4)

    def latency_func_bypass(x):
        return 0.

    latency = np.array([])
    for i in range(len(x)):
        if link_type[i] == 0:
            latency = np.append(latency, latency_func_charging(
                x[i], t_0[i], capacity[i], J))
        elif link_type[i] == 1:
            latency = np.append(latency, latency_func_regular(
                x[i], t_0[i], capacity[i]))
        elif link_type[i] == 2:
            latency = np.append(latency, latency_func_bypass(x[i]))
    return latency


def get_metrics(ue_gaso, ue, t_0, c, link_type, J):
    def latency_func(x, t_0, c):
        return t_0 * (1. + 0.15 * (x / c) ** 4)

    def emission_func(t_0, latency):
        return 0.2038 * latency * np.math.exp(0.7962 * t_0 / latency)

    emission = np.array([])
    for i in range(len(ue)):
        latency = latency_func(ue[i], t_0[i], c[i])
        emission = np.append(emission, emission_func(t_0[i], latency))
    
    latency_arr = get_latency_with_all_types(ue, link_type, t_0, c, J)

    total_emission = np.sum(ue_gaso * emission)
    total_latency = np.sum(ue * latency_arr)

    return total_emission, total_latency


# q_gaso = np.array([[30, 25],
#                    [20, 25],
#                    [30, 20],
#                    [20, 20],
#                    [30, 25],
#                    [15, 15],
#                    [15, 15]])

# q_elec = np.array([[15, 10],
#                    [10, 5],
#                    [10, 15],
#                    [10, 10],
#                    [15, 10],
#                    [5, 5],
#                    [5, 5]])

q_gaso = np.array([[30, 25],
                   [30, 25],
                   [30, 25],
                   [30, 25],
                   [30, 25],
                   [30, 25],
                   [30, 25]])

q_elec = np.array([[15, 10],
                   [15, 10],
                   [15, 10],
                   [15, 10],
                   [15, 10],
                   [15, 10],
                   [15, 10]])

greedy_week_emission = np.array([])
greedy_week_latency = np.array([])
greedy_week_opf_cost = np.array([])
greedy_week_toll = np.array([])
greedy_week_ue = np.array([])
greedy_week_lmp = np.array([])
# 第一天
toll_init = np.zeros(n_l)
ue_init, delta_gaso, delta_elec, ue_gaso, ue_elec = adaptive_path_generation(num_link=n_l, num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v,
                                                                            t_0=t_0, capacity=c, lmp=lmp_init, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso[0], q_elec=q_elec[0], tau=toll_init,
                                                                            verbose=False)
best_toll, ue, ue_gaso, f, lmp = greedy_genetic_algorithm(ue_prev=ue_init, num_link=12, x_min=0, x_max=10, decimal=3, population_size=30, GA_MAX_ITER=50,
                                    epsilon=0.5, BRD_MAX_ITER=10, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                    p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
                                    num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v, t_0=t_0, capacity=c, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso[0], q_elec=q_elec[0])

emiss, late = get_metrics(ue_gaso, ue, t_0, c, link_type_v, 0.05)
greedy_week_emission = np.append(greedy_week_emission, emiss)
greedy_week_latency = np.append(greedy_week_latency, late)
greedy_week_opf_cost = np.append(greedy_week_opf_cost, f)
greedy_week_toll = np.array(best_toll)
greedy_week_ue = np.array(ue)
greedy_week_lmp = np.array(lmp)

# 6天循环
for i in range(1, 7):
    best_toll, ue, ue_gaso, f, lmp = greedy_genetic_algorithm(ue_prev=ue, num_link=12, x_min=0, x_max=10, decimal=3, population_size=30, GA_MAX_ITER=50,
                                    epsilon=0.5, BRD_MAX_ITER=10, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                    p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
                                    num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v, t_0=t_0, capacity=c, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso[i], q_elec=q_elec[i])
    emiss, late = get_metrics(ue_gaso, ue, t_0, c, link_type_v, 0.05)
    greedy_week_emission = np.append(greedy_week_emission, emiss)
    greedy_week_latency = np.append(greedy_week_latency, late)
    greedy_week_opf_cost = np.append(greedy_week_opf_cost, f)
    greedy_week_toll = np.vstack((greedy_week_toll, best_toll))
    greedy_week_ue = np.vstack((greedy_week_ue, ue))
    greedy_week_lmp = np.vstack((greedy_week_lmp, lmp))

np.savetxt('results/greedy/emission.txt', greedy_week_emission)
np.savetxt('results/greedy/latency.txt', greedy_week_latency)
np.savetxt('results/greedy/opf_cost.txt', greedy_week_opf_cost)
np.savetxt('results/greedy/toll.txt', greedy_week_toll)
np.savetxt('results/greedy/ue.txt', greedy_week_ue)
np.savetxt('results/greedy/lmp.txt', greedy_week_lmp)


from gab_road_emission_pricing import genetic_algorithm
from best_response_decomposition import best_response_decomposition

utopia_week_emission = np.array([])
utopia_week_latency = np.array([])
utopia_week_opf_cost = np.array([])
utopia_week_toll = []
utopia_week_ue = []
utopia_week_lmp = []

for i in range(7):
    best_toll, best_fitness, fitness_iter = genetic_algorithm(num_link=12, x_min=0, x_max=10, decimal=3, population_size=30, GA_MAX_ITER=50,
                                    epsilon=0.5, BRD_MAX_ITER=10, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                    p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
                                    num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v, t_0=t_0, capacity=c, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso[i], q_elec=q_elec[i])

    ue, ue_gaso, lmp, DELTA, f, p_g, P, q_g, Q, U = best_response_decomposition(epsilon=0.5, MAX_ITER=10, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                      p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=best_toll,
                                                      num_link=12, num_od=2, Lambda=L, I_rs=I_rs, charging_flag=d, omega=10, link_type=link_type_v, t_0=t_0, capacity=c, J=0.05, energy_demand=energy_demand, q_gaso=q_gaso[i], q_elec=q_elec[i],
                                                      verbose=False)
    emiss, late = get_metrics(ue_gaso, ue, t_0, c, link_type_v, 0.05)
    utopia_week_emission = np.append(utopia_week_emission, emiss)
    utopia_week_latency = np.append(utopia_week_latency, late)
    utopia_week_opf_cost = np.append(utopia_week_opf_cost, f)
    utopia_week_toll.append(best_toll)
    utopia_week_ue.append(ue)
    utopia_week_lmp.append(lmp)

np.savetxt('results/utopia/emission.txt', utopia_week_emission)
np.savetxt('results/utopia/latency.txt', utopia_week_latency)
np.savetxt('results/utopia/opf_cost.txt', utopia_week_opf_cost)
np.savetxt('results/utopia/toll.txt', np.array(utopia_week_toll))
np.savetxt('results/utopia/ue.txt', np.array(utopia_week_ue))
np.savetxt('results/utopia/lmp.txt', np.array(utopia_week_lmp))
