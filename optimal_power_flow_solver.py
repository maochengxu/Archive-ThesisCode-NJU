import numpy as np
import cvxpy as cp


def opf_obj(num_bus, a, b, rho, P: cp.Variable, p: cp.Variable, pi_0):
    sum = 0.
    for j in range(num_bus):
        sum = sum + (a * p[j] ** 2 + b * p[j])
    return sum + rho * pi_0 @ P


def opf_solver(num_bus: float, num_line: float, a, b, rho, pi, child_bus, r, x,
               p_dt: float, p_dc: np.ndarray, q_d: float, U_0: float, p_gn, p_gm, q_gn, q_gm, U_n, U_m, S):
    # Define Variables
    P = cp.Variable(num_line, nonneg=True)
    Q = cp.Variable(num_line)
    I = cp.Variable(num_line, nonneg=True)
    U = cp.Variable(num_bus, nonneg=True)
    p_g = cp.Variable(num_bus, nonneg=True)
    q_g = cp.Variable(num_bus)

    # Define objective
    obj = cp.Minimize(opf_obj(num_bus, a, b, rho, P, p_g, pi))

    # Define Constraints
    # Cons-CBPF
    cbpf_1 = [P[0] + p_g[0] - r[0] * I[0] == child_bus[0] @ P + p_dt + p_dc[0]]
    cbpf_2 = [Q[0] + q_g[0] - x[0] * I[0] == child_bus[0] @ Q + q_d]
    cbpf_3 = [U[0] == U_0 - 2 *
              (r[0] * P[0] + x[0] * Q[0]) + (r[0] ** 2 + x[0] ** 2) * I[0]]
    cbpf_4 = [cp.SOC(I[0] + U_0, cp.hstack([2 * P[0], 2 * Q[0], I[0] - U_0]))]

    for l in range(1, num_line):
        cbpf_1.append(P[l] + p_g[l] - r[l] * I[l] ==
                      child_bus[l] @ P + p_dt + p_dc[l])
        cbpf_2.append(Q[l] + q_g[l] - x[l] * I[l] == child_bus[l] @ Q + q_d)
        i = np.nonzero(child_bus[:, l])[0][0]
        cbpf_3.append(U[l] == U[i] - 2 * (r[l] * P[l] + x[l]
                      * Q[l]) + (r[l] ** 2 + x[l] ** 2) * I[l])
        cbpf_4_tmp = cp.SOC(
            I[l] + U[i], cp.hstack([2 * P[l], 2 * Q[l], I[l] - U[i]]))
        cbpf_4.append(cbpf_4_tmp)

    # Cons-BND
    bnd_1 = []
    bnd_2 = []
    bnd_3 = []
    bnd_4 = []
    bnd_5 = []
    bnd_6 = []

    for j in range(num_bus):
        bnd_1.append(p_g[j] <= p_gm)
        bnd_1.append(p_g[j] >= p_gn)
        bnd_2.append(q_g[j] <= q_gm)
        bnd_2.append(q_g[j] >= q_gn)
        bnd_3.append(U[j] <= U_m)
        bnd_3.append(U[j] >= U_n)

    for l in range(num_line):
        bnd_4.append(P[l] ** 2 + Q[l] ** 2 <= S ** 2)
        bnd_5.append(P[l] - r[l] * I[l] >= 0)
        bnd_6.append(Q[l] - x[l] * I[l] >= 0)

    constrs = tuple(cbpf_1 + cbpf_2 + cbpf_3 + cbpf_4 +
                    bnd_1 + bnd_2 + bnd_3 + bnd_4 + bnd_5 + bnd_6)

    prob = cp.Problem(obj, constraints=constrs)
    f = prob.solve(solver=cp.MOSEK, verbose=False)
    lmp = np.array([np.abs(constrs[i].dual_value) for i in range(num_line)])
    p_g_value = p_g.value
    P_value = P.value
    q_g_value = q_g.value
    Q_value = Q.value
    U_value = U.value
    return lmp, p_g_value, P_value, q_g_value, Q_value, U_value, f
