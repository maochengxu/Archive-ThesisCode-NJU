import numpy as np
from tqdm import tqdm
import pandas as pd
from best_response_decomposition import best_response_decomposition
import matplotlib.pyplot as plt


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


def fitness_func(ue_gaso, ue, t_0, c, link_type, J):
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

    return 20 - (0.7 * total_emission / 100 + 0.3 * total_latency / 300)


def chromo_length(x_min: float, x_max: float, decimal: int) -> int:
    chromo_len = np.math.log2(1 + (x_max - x_min) * 10 ** decimal)
    chromo_len = np.math.ceil(chromo_len)
    return chromo_len


def encoding_chromo(x: float, x_min, x_max, chromo_len) -> list:
    max_binary = '0b' + ''.join(['1' for _ in range(chromo_len)])
    max_num = np.int(max_binary, base=0)
    pi = (x_max - x_min) / max_num
    index = int(round((x - x_min) / pi))
    binary = bin(index)[2:]
    num_zero = chromo_len - len(binary)
    chromo = [0 for _ in range(num_zero)] + [eval(i) for i in binary]
    return chromo


def decoding_chromo(chromo: list, x_min, x_max, decimal) -> float:
    chromo_len = len(chromo)
    tmp_decode = np.int('0b' + ''.join(str(i) for i in chromo), base=0)
    x = x_min + (x_max - x_min) * np.sum(tmp_decode) / (2 ** chromo_len - 1)
    return round(x, decimal)


def get_accu_prob_array(fitness_array):
    # def normalization(data):
    #     _range = np.max(data) - np.min(data)
    #     return (data - np.min(data)) / _range

    # max_fitness = np.max(fitness_array)
    # fitness_array_norm = normalization(fitness_array)
    fitness_sum = np.sum(fitness_array)
    prob_array = fitness_array / fitness_sum
    accu_prob_array = np.array([prob_array[0]])
    for i in range(1, len(prob_array)):
        accu_prob_array = np.append(
            accu_prob_array, accu_prob_array[i-1] + prob_array[i])

    return accu_prob_array


def reproduction(accu_prob_array, fitness_array):
    rnd = np.random.uniform(0, 1, 2)
    index_set = [0, 0]
    for i in range(len(accu_prob_array)):
        if rnd[0] < accu_prob_array[i]:
            index_set[0] = i
            break
    for i in range(len(accu_prob_array)):
        if rnd[1] < accu_prob_array[i]:
            index_set[1] = i
            break
    if fitness_array[index_set[0]] >= fitness_array[index_set[1]]:
        return index_set[0]
    else:
        return index_set[1]


def crossover(prob_cross, prob_muta, num_link, fitness_array, population: np.ndarray, x_min, x_max, decimal, chromo_len,
              epsilon, BRD_MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
              p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S,
              num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
              capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray):
    
    dice_cross = np.random.uniform(0, 1)
    if dice_cross <= prob_cross:
        for _ in range(5):
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

            ue_1, ue_gaso_1, lmp_1, D_1, f_1, p_g_1, P_1, q_g_1, Q_1, U_1 = best_response_decomposition(epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                            p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll_child_1,
                                                            num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
            fitness_child_1 = fitness_func(ue_gaso_1, ue_1, t_0, capacity, link_type, J)
            ue_2, ue_gaso_2, lmp_2, D_2, f_2, p_g_2, P_2, q_g_2, Q_2, U_2 = best_response_decomposition(epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
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

            # population[min_arr[0]] = child_1
            # fitness_array[min_arr[0]] = fitness_child_1
            # population[min_arr[1]] = child_2
            # fitness_array[min_arr[1]] = fitness_child_2

    return population, fitness_array


# def mutation(prob, population: np.ndarray):
# def mutation(prob, num_link, fitness_array, population: np.ndarray, x_min, x_max, decimal, chromo_len,
#               epsilon, BRD_MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
#               p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S,
#               num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
#               capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray):
#     dice = np.random.uniform(0, 1)
#     if dice <= prob:
#         muta_index = int(population.shape[0] * np.random.uniform(0, 1))
#         muta_position = int(population.shape[1] * np.random.uniform(0, 1))
#         population[muta_index][muta_position] = not population[muta_index][muta_position]
#         mutated_chromo = population[muta_index]

#         toll_mutated = np.zeros(num_link)
#         for j in range(num_link):
#             x_tmp = decoding_chromo(mutated_chromo[(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
#             toll_mutated[j] = x_tmp
        
#         ue, ue_gaso, lmp, D = best_response_decomposition(epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
#                                                           p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll_mutated,
#                                                           num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
#         fitness_mutated = fitness_func(ue_gaso, ue, t_0, capacity)
#         fitness_array[muta_index] = fitness_mutated
#     return population, fitness_array


def random_init(num_link, x_min, x_max, population_size, chromo_len):
    chromo_tmp = []
    for i in range(num_link):
        x_tmp = np.random.uniform(x_min, x_max)
        chromo_tmp = chromo_tmp + \
            encoding_chromo(x_tmp, x_min, x_max, chromo_len)
    population = np.array(chromo_tmp)
    for i in range(population_size - 1):
        chromo_tmp = []
        for j in range(num_link):
            x_tmp = np.random.uniform(x_min, x_max)
            chromo_tmp = chromo_tmp + \
                encoding_chromo(x_tmp, x_min, x_max, chromo_len)
        population = np.vstack((population, np.array(chromo_tmp)))
    return population


# def select_best(population, x_min, x_max, decimal, num_link, chromo_len,
#                 epsilon, BRD_MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
#                 p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S,
#                 num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
#                 capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray):
#     fitness_array = np.array([])
#     for i in range(len(population)):
#         toll = np.zeros(num_link)
#         for j in range(num_link):
#             x_tmp = decoding_chromo(
#                 population[i][(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
#             toll[j] = x_tmp
#         ue, ue_gaso, lmp, D = best_response_decomposition(epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
#                                                           p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll,
#                                                           num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
#         fitness = fitness_func(ue_gaso, ue, t_0, capacity)
#         fitness_array = np.append(fitness_array, fitness)
#     max_index = np.argmax(fitness_array)
#     best_toll = np.zeros(num_link)
#     for j in range(num_link):
#         x_tmp = decoding_chromo(population[max_index][(
#             j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
#         best_toll[j] = x_tmp
#     return best_toll, fitness_array[max_index]


def genetic_algorithm(num_link, x_min, x_max, decimal, population_size, GA_MAX_ITER,
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
        ue, ue_gaso, lmp, D, f, p_g, P, q_g, Q, U = best_response_decomposition(epsilon=epsilon, MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
                                                          p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S, emission_toll=toll,
                                                          num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
        fitness = fitness_func(ue_gaso, ue, t_0, capacity, link_type, J)
        fitness_array = np.append(fitness_array, fitness)

    for i in tqdm(range(GA_MAX_ITER)):
        population, fitness_array = crossover(
            0.95, 0.033, num_link, fitness_array, population, x_min, x_max, decimal, chromo_len,
            epsilon=epsilon, BRD_MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
            p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
            num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)
        
        # population, fitness_array = mutation(
        #     0.033, num_link, fitness_array, population, x_min, x_max, decimal, chromo_len,
        #     epsilon=epsilon, BRD_MAX_ITER=BRD_MAX_ITER, num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x, p_dt=p_dt, q_d=q_d, U_0=U_0,
        #     p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S,
        #     num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type, t_0=t_0, capacity=capacity, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec)

        max_fitness = np.max(fitness_array)
        fitness_iter.append(max_fitness)
        print('Iteration ' + str(i) + ': ' + str(max_fitness))

    max_index = np.argmax(fitness_array)
    best_fitness = fitness_array[max_index]
    best_toll = np.zeros(num_link)
    for j in range(num_link):
        x_tmp = decoding_chromo(population[max_index][(j*chromo_len):((j+1)*chromo_len)], x_min, x_max, decimal)
        best_toll[j] = x_tmp
    return best_toll, best_fitness, fitness_iter
