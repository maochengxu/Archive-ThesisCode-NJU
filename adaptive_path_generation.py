import numpy as np
import cvxpy as cp
import pandas as pd
from tqdm import tqdm


def latency_func(x: np.ndarray, link_type: np.ndarray, t_0: np.ndarray, capacity: np.ndarray, J: float) -> np.ndarray:
    """Calculate the latency for all the links.

    Args:
        x (np.ndarray): The vector denoting the flow on every link.
        link_type (np.ndarray): The vector denoting the type of each link. 0 for charging link, 1 for regular link and 2 for bypass link.
        t_0 (np.ndarray): The free travel time/charging time of each link.
        capacity (np.ndarray): The capacity of each link.
        J (float): The coefficient in equation (18).

    Returns:
        np.ndarray: Value of the latency function for each link.
    """
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


def cal_charging_cost(lmp: np.ndarray, link_type: np.ndarray, energy_demand: float) -> np.ndarray:
    """Calculate the charging cost.

    Args:
        lmp (np.ndarray): The locational marginal price, aka lambda_a^j.
        link_type (np.ndarray): The vector denoting the type of each link. 0 for charging link, 1 for regular link and 2 for bypass link.
        energy_demand (float): E_B

    Returns:
        np.ndarray: The charging cost.
    """
    charging_cost = np.array([])
    for i in range(len(link_type)):
        if link_type[i] == 0:
            charging_cost = np.append(charging_cost, lmp[i] * energy_demand)
        else:
            charging_cost = np.append(charging_cost, 0.)
    return charging_cost


def travel_expense(x: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray, capacity: np.ndarray, lmp: np.ndarray, J: float, energy_demand: float, x_gaso, tau) -> tuple:
    """The travel expense for GVs and EVs

    Args:
        x (np.ndarray): The vector denoting the flow on every link.
        omega (float): The monetary value of time.
        link_type (np.ndarray): The vector denoting the type of each link. 0 for charging link, 1 for regular link and 2 for bypass link.
        t_0 (np.ndarray): The free travel time/charging time of each link.
        capacity (np.ndarray): The capacity of each link.
        lmp (np.ndarray): The locational marginal price, aka lambda_a^j.
        J (float): The coefficient in equation (18).
        energy_demand (float): E_B

    Returns:
        tuple: travel expense for GV/EV
    """
    # emission_array = np.zeros((len(x), ))
    # for i in range(len(x)):
    #     latency = t_0[i] * (1. + 0.15 * (x[i] / capacity[i]) ** 4)
    #     emission_array[i] = 0.2038 * latency * np.math.exp(0.7962 * t_0[i] / latency)
    # p_gaso = omega * latency_func(x, link_type, t_0, capacity, J) + np.sum(tau * emission_array * x_gaso)
    # p_gaso = omega * latency_func(x, link_type, t_0, capacity, J) + tau @ x_gaso
    p_gaso = omega * latency_func(x, link_type, t_0, capacity, J) + tau
    p_elec = omega * latency_func(x, link_type, t_0, capacity,
                                  J) + cal_charging_cost(lmp, link_type, energy_demand)
    return p_gaso, p_elec


def path_solver(num_link: int, p: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray,
                Lambda: np.ndarray, vehicle_type: bool) -> tuple:
    """Solve one MILP to find one possible path for EV or GV seperately.

    Args:
        num (int): The total number of links.
        p (np.ndarray): Travel expense vector.
        I_rs (np.ndarray): Indicator of O-D pair.
        charging_flag (np.ndarray): Indicator of wheather one link is a charging link.
        Lambda (np.ndarray): The node-link incidence matrix.
        vehicle_type (bool): True for EV and False for GV.

    Returns:
        tuple: Total cost of the path, path
    """
    v = cp.Variable(num_link, boolean=True)
    constr1 = Lambda @ v == I_rs
    if vehicle_type == False:
        constr2 = charging_flag @ v == 0
        obj = cp.Minimize(p @ v)
        prob = cp.Problem(obj, (constr1, constr2))
    elif vehicle_type == True:
        constr2 = charging_flag @ v == 1
        obj = cp.Minimize(p @ v)
        prob = cp.Problem(obj, (constr1, constr2))

    u = prob.solve(solver=cp.MOSEK)
    path = v.value

    return u, path


def tap_solver(num_link: int, num_od: int,  omega: float, J: float, t_0: np.ndarray, energy_demand: float, c: np.ndarray,
               lmp: np.ndarray, link_type: np.ndarray, Delta_gaso: list, Delta_elec: list, q_gaso: np.ndarray, q_elec: np.ndarray, charging_flag, verbose, tau) -> np.ndarray:
    """Solve the Traffic Assignment Problem

    Args:
        num_link (int): The total number of links.
        num_od (int): The total number of ods.
        omega (float): The monetary value of time.
        J (float): The coefficient in equation (18).
        t_0 (np.ndarray): The free travel time/charging time of each link.
        energy_demand (float): E_B
        c (np.ndarray): The capacity of each link.
        lmp (np.ndarray): The locational marginal price, aka lambda_a^j.
        link_type (np.ndarray): The vector denoting the type of each link. 0 for charging link, 1 for regular link and 2 for bypass link.
        Delta_gaso (list): The link-path incidence matrix for GVs.
        Delta_elec (list): The link-path incidence matrix for EVs.
        q_gaso (np.ndarray): Trip rate of GV.
        q_elec (np.ndarray): Trip rate of EV.

    Returns:
        np.ndarray: The user equilibrium.
    """
    x = cp.Variable(num_link)
    x_gaso = cp.Variable(num_link)
    x_elec = cp.Variable(num_link)
    f_gaso = []
    f_elec = []
    for i in range(num_od):
        f_gaso.append(cp.Variable(Delta_gaso[i].shape[1], nonneg=True))
        f_elec.append(cp.Variable(Delta_elec[i].shape[1], nonneg=True))

    constr1 = []
    constr2 = []
    constr3 = []
    constr4 = []
    # constr5 = []
    gaso_tmp = 0.
    elec_tmp = 0.
    for i in range(num_od):
        # constr_tmp = constr_tmp + \
        #     Delta_gaso[i] @ f_gaso[i] + Delta_elec[i] @ f_elec[i]
        gaso_tmp = gaso_tmp + Delta_gaso[i] @ f_gaso[i]
        elec_tmp = elec_tmp + Delta_elec[i] @ f_elec[i]
        constr2.append(cp.sum(f_gaso[i]) == q_gaso[i])
        constr3.append(cp.sum(f_elec[i]) == q_elec[i])
        # constr4.append(f_gaso[i] >= 0)
        # constr5.append(f_elec[i] >= 0)
    constr1.append(x_gaso == gaso_tmp)
    constr1.append(x_elec == elec_tmp)
    constr1.append(x == x_gaso + x_elec)
    constr4.append(cp.diag(cp.reshape(x, (num_link, 1)) @ charging_flag.reshape((1, -1)))
                   <= cp.diag(c.reshape((-1, 1)) @ charging_flag.reshape((1, -1))))

    obj = cp.Minimize(
        tap_obj(omega, J, t_0, energy_demand, x, c, lmp, link_type, x_gaso, tau))

    constrs = tuple(constr1 + constr2 + constr3 + constr4)

    prob = cp.Problem(obj, constraints=constrs)
    f = prob.solve(solver=cp.MOSEK, verbose=verbose, qcp=False)
    ue = x.value
    ue_gaso = x_gaso.value
    ue_elec = x_elec.value
    return ue, ue_gaso, ue_elec


def tap_obj(omega: float, J: float, t_0: np.ndarray, energy_demand: float, x: cp.Variable,
            c: np.ndarray, lmp: np.ndarray, link_type: np.ndarray, x_gaso, tau) -> float:
    """The objective function of the TAP problem.

    Args:
        omega (float): The monetary value of time.
        J (float): The coefficient in equation (18).
        t_0 (np.ndarray): The free travel time/charging time of each link.
        energy_demand (float): E_B
        x (cp.Variable): The desisive variable.
        c (np.ndarray): The capacity of each link.
        lmp (np.ndarray): The locational marginal price, aka lambda_a^j.
        link_type (np.ndarray): The vector denoting the type of each link. 0 for charging link, 1 for regular link and 2 for bypass link.

    Returns:
        float: The total cost.
    """
    cost = 0.
    for i in range(len(link_type)):
        if link_type[i] == 0:
            tmp = t_0[i] * (1 - J) * x[i] + t_0[i] * c[i] * J * \
                (cp.log(c[i]) - cp.log((c[i] - x[i])))
            cost_tmp = omega * tmp + lmp[i] * energy_demand * x[i]
            cost = cost + cost_tmp
        elif link_type[i] == 1:
            tmp = t_0[i] * (x[i] + 0.03 * x[i] ** 5 / (c[i] ** 4))
            cost_tmp = omega * tmp
            cost = cost + cost_tmp + tau[i] * x_gaso[i]
        elif link_type[i] == 2:
            cost = cost + 0.
    return cost


def update_delta(v: np.ndarray, delta: np.ndarray, vehicle_type: bool) -> np.ndarray:
    """Update the link-path incidence matrix.

    Args:
        v (np.ndarray): The new path.
        delta (np.ndarray): The original incidence matrix.
        vehicle_type (bool): True for EV and False for GV.

    Returns:
        np.ndarray: The new incidence matrix.
    """
    # Check if duplicated
    for i in range(np.shape(delta)[1]):
        diff = np.linalg.norm(v - delta[:, i])
        if diff <= 1e-6:
            return delta
    v_p = np.zeros(len(v))
    for i in range(len(v)):
        if v[i] < 0.5:
            v_p[i] = 0.
        else:
            v_p[i] = 1.
    v = v_p.reshape((-1, 1))
    delta = np.append(delta, v, axis=1)

    return delta


def compare_u(u1: np.ndarray, u2: np.ndarray) -> bool:
    """Compare two travel expense vectors

    Args:
        u1 (np.ndarray): The first tev
        u2 (np.ndarray): The second tev

    Returns:
        bool: Result
    """
    flag = 1
    for i in range(len(u1)):
        if u1[i] < u2[i]:
            flag = 0
    if flag == 0:
        return False
    elif flag == 1:
        return True


def adaptive_path_generation(num_link: int, num_od: int, Lambda: np.ndarray, I_rs: np.ndarray, charging_flag: np.ndarray, omega: float, link_type: np.ndarray, t_0: np.ndarray,
                             capacity: np.ndarray, lmp: np.ndarray, J: float, energy_demand: float, q_gaso: np.ndarray, q_elec: np.ndarray, tau, MAX_ITER: float = 20, verbose: bool = False) -> tuple:
    """Calculate the user equilibrium.

    Args:
        num_link (int): The total number of links.
        num_od (int): The total number of ods.
        Lambda (np.ndarray): The node-link incidence matrix.
        I_rs (np.ndarray): Indicator of O-D pair.
        charging_flag (np.ndarray): Indicator of wheather one link is a charging link.
        omega (float): The monetary value of time.
        link_type (np.ndarray): The vector denoting the type of each link. 0 for charging link, 1 for regular link and 2 for bypass link.
        t_0 (np.ndarray): The free travel time/charging time of each link.
        capacity (np.ndarray): The capacity of each link.
        lmp (np.ndarray): The locational marginal price, aka lambda_a^j.
        J (float): The coefficient in equation (18).
        energy_demand (float): E_B
        q_gaso (np.ndarray): Trip rate of GV.
        q_elec (np.ndarray): Trip rate of EV.

    Returns:
        tuple: The user equilibrium (the aggregated traffic flow), Final link-path incidence matrix for GVs, Final link-path incidence matrix for EVs.
    """
    # Initialize
    aggregated_traffic_flow = np.zeros((num_link, ))
    x_gaso = np.zeros((num_link, ))
    p_gaso, p_elec = travel_expense(
        aggregated_traffic_flow, omega, link_type, t_0, capacity, lmp, J, energy_demand, x_gaso, tau)
    # Find initial paths for GV and EV
    u_gaso = np.array([])
    u_elec = np.array([])
    delta_gaso = []
    delta_elec = []
    for i in range(num_od):
        u_gaso_rs, path_gaso_rs = path_solver(
            num_link=num_link, p=p_gaso, I_rs=I_rs[:, i], charging_flag=charging_flag, Lambda=Lambda, vehicle_type=False)
        u_elec_rs, path_elec_rs = path_solver(
            num_link=num_link, p=p_elec, I_rs=I_rs[:, i], charging_flag=charging_flag, Lambda=Lambda, vehicle_type=True)
        # Build initial link-path incidence matrixes.
        delta_gaso_rs = np.array(path_gaso_rs).reshape((-1, 1))
        delta_elec_rs = np.array(path_elec_rs).reshape((-1, 1))
        delta_gaso_rs = update_delta(
            v=path_gaso_rs, delta=delta_gaso_rs, vehicle_type=False)
        delta_elec_rs = update_delta(
            v=path_elec_rs, delta=delta_elec_rs, vehicle_type=True)
        u_gaso = np.append(u_gaso, u_gaso_rs)
        u_elec = np.append(u_elec, u_elec_rs)
        delta_gaso.append(delta_gaso_rs)
        delta_elec.append(delta_elec_rs)

    for _ in range(MAX_ITER):
        # Solve TAP
        try:
            ue, ue_gaso, ue_elec = tap_solver(num_link=num_link, num_od=num_od, omega=omega, J=J, t_0=t_0, energy_demand=energy_demand, c=capacity, lmp=lmp,
                                              link_type=link_type, Delta_gaso=delta_gaso, Delta_elec=delta_elec, q_gaso=q_gaso, q_elec=q_elec, charging_flag=charging_flag, verbose=verbose, tau=tau)
        except:
            ue = None
            ue_gaso = ue_elec = None

        if ue is not None:
            aggregated_traffic_flow = ue
        else:
            aggregated_traffic_flow = np.zeros((num_link, ))
            ue_gaso = np.zeros((num_link, ))
            for i in range(num_od):
                approx_flow_gaso = q_gaso[i] / delta_gaso[i].shape[1]
                approx_flow_elec = q_elec[i] / delta_elec[i].shape[1]
                tmp = np.sum(delta_gaso[i] * approx_flow_gaso, axis=1) + \
                    np.sum(delta_elec[i] * approx_flow_elec, axis=1)
                tmp_gaso = np.sum(delta_gaso[i] * approx_flow_gaso, axis=1)
                aggregated_traffic_flow = aggregated_traffic_flow + \
                    tmp.reshape((-1, ))
                ue_gaso = ue_gaso + tmp_gaso.reshape((-1, ))
        # Update travel expenses
        p_gaso, p_elec = travel_expense(
            aggregated_traffic_flow, omega, link_type, t_0, capacity, lmp, J, energy_demand, ue_gaso, tau)
        # Find minimal travel expenses for each OD pair
        u_opt_gaso = np.array([])
        u_opt_elec = np.array([])
        path_gaso_new = []
        path_elec_new = []
        for i in range(num_od):
            u_gaso[i] = np.min(p_gaso @ delta_gaso[i])
            u_elec[i] = np.min(p_elec @ delta_elec[i])
            # Solve the MILPs to find optimal solution (build new paths)
            u_opt_gaso_rs, path_gaso_rs = path_solver(
                num_link=num_link, p=p_gaso, I_rs=I_rs[:, i], charging_flag=charging_flag, Lambda=Lambda, vehicle_type=False)
            u_opt_elec_rs, path_elec_rs = path_solver(
                num_link=num_link, p=p_elec, I_rs=I_rs[:, i], charging_flag=charging_flag, Lambda=Lambda, vehicle_type=True)
            u_opt_gaso = np.append(u_opt_gaso, u_opt_gaso_rs)
            u_opt_elec = np.append(u_opt_elec, u_opt_elec_rs)
            path_gaso_new.append(path_gaso_rs)
            path_elec_new.append(path_elec_rs)
        # Break conditions
        flag_gaso = compare_u(u_opt_gaso, u_gaso)
        flag_elec = compare_u(u_opt_elec, u_elec)

        if flag_gaso and flag_elec:
            return aggregated_traffic_flow, delta_gaso, delta_elec, ue_gaso, ue_elec

        if not flag_gaso:
            for i in range(num_od):
                if u_opt_gaso[i] < u_gaso[i]:
                    delta_gaso[i] = update_delta(
                        v=path_gaso_new[i], delta=delta_gaso[i], vehicle_type=False)

        if not flag_elec:
            for i in range(num_od):
                if u_opt_elec[i] < u_elec[i]:
                    delta_elec[i] = update_delta(
                        v=path_elec_new[i], delta=delta_elec[i], vehicle_type=True)
    return aggregated_traffic_flow, delta_gaso, delta_elec, ue_gaso, ue_elec
