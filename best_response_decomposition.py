import numpy as np
import pandas as pd
from adaptive_path_generation import adaptive_path_generation
from optimal_power_flow_solver import opf_solver


def load_injection(ue, charging_flag, energy_demand):
    p_dc = np.array([])
    for i in range(len(ue)):
        if charging_flag[i] == 1:
            p_dc = np.append(p_dc, ue[i] * energy_demand * 100)
    return p_dc


def best_response_decomposition(epsilon, MAX_ITER, num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
                                p_dt: float, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S, emission_toll,
                                num_link: int, num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
                                capacity: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray, verbose=False):
    # Initialization
    DELTA = []
    ue = np.zeros((num_link, ))
    p_dc = load_injection(ue, charging_flag, energy_demand)
    lmp, p_g, P, q_g, Q, U, f = opf_solver(num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x,
                          p_dt=p_dt, p_dc=p_dc, q_d=q_d, U_0=U_0, p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S)
    lmp_all = np.zeros(num_link)
    j_ = 0
    for i in range(num_link):
        if link_type[i] == 0:
            lmp_all[i] = lmp[j_]
            j_ = j_ + 1

    for k in range(MAX_ITER):
        ue_new, delta_gaso, delta_elec, ue_gaso, ue_elec = adaptive_path_generation(num_link=num_link, num_od=num_od, Lambda=Lambda, I_rs=I_rs, charging_flag=charging_flag, omega=omega, link_type=link_type,
                                                                                    t_0=t_0, capacity=capacity, lmp=lmp_all, J=J, energy_demand=energy_demand, q_gaso=q_gaso, q_elec=q_elec, tau=emission_toll,
                                                                                    verbose=verbose)
        DELTA_TN = np.linalg.norm(ue_new - ue)
        ue = ue_new
        # p_dc = load_injection(ue, charging_flag, energy_demand)[:4] + load_injection(ue, charging_flag, energy_demand)[4:]
        p_dc = load_injection(ue, charging_flag, energy_demand)
        lmp_new, p_g, P, q_g, Q, U, f = opf_solver(num_bus=num_bus, num_line=num_line, a=a, b=b, rho=rho, pi=pi, child_bus=child_bus, r=r, x=x,
                              p_dt=p_dt, p_dc=p_dc, q_d=q_d, U_0=U_0, p_gn=p_gn, p_gm=p_gm, q_gn=q_gn, q_gm=q_gm, U_n=U_n, U_m=U_m, S=S)
        DELTA_PDN = np.linalg.norm(lmp_new - lmp)
        lmp = lmp_new
        lmp_all = np.zeros(num_link)
        j_ = 0
        for i in range(num_link):
            if link_type[i] == 0:
                lmp_all[i] = lmp[j_]
                j_ = j_ + 1

        DELTA.append(DELTA_TN + DELTA_PDN)

        if DELTA_TN + DELTA_PDN <= epsilon:
            return ue, ue_gaso, lmp, DELTA, f, p_g, P, q_g, Q, U
    return ue, ue_gaso, lmp, DELTA, f, p_g, P, q_g, Q, U


