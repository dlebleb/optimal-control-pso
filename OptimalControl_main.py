import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from helper_funcs import animate_trajectory

#min. effort kontrol
#Toplam state vektoru 8-boyutlu.
#Toplam control vektoru 4-boyutlu.

#z=[x1​,…,x10​,u0​,…,u9​]

#rviz de +x(ileri/kirmizi), +y(sol/yesil)

# -----------------------------
# Problem size
# -----------------------------
N = 20        # number of intervals
nx = 8        # state dimension
nu = 4        # control dimension
tf = 10 
#birbirine yaklasma hizina gore alacaksin tf degerini, otekinin hizinin senin hizina projection'i aldigin zaman, birbirine yaklasma hizi cikiyor. v1.v2costheta, initial mesafe de belli, mesafe/hiz = tf'in ne olacagi cikar. 
#min tf ve max tf verebilirsin, bunlar arasinda olsun. 2-15 saniye arasinda sinirlayabilirsin.
dt = tf/N
#n_dec = N * nx + N * nu
n_dec = N * nu

mw = 100
mo = 80


# State:
# x = [p1x, p1y, p2x, p2y, v1x, v1y, v2x, v2y]^T
#
# Control:
# u = [u1x, u1y, u2x, u2y]^T

# -----------------------------
# Continuous-time dynamics
# x_dot = A x + B u
# -----------------------------
A = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)

B = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1/mw, 0, 0, 0],
    [0, 1/mw, 0, 0],
    [0, 0, 1/mo, 0],
    [0, 0, 0, 1/mo],
], dtype=float)


# -----------------------------
# Cost matrices / weights / will be tuned later
# -----------------------------
D = np.eye(nu)      # positive definite diagonal control matrix

# Q1 = 50*np.eye(2)      # terminal cost for agent 1: [p1x, p1y, v1x, v1y]
# Q2 = 50*np.eye(2)      # terminal cost for agent 2: [p2x, p2y, v2x, v2y]

Q1 = 500*np.eye(2)
Q2 = 500*np.eye(2)

alpha = 1.0         # weight for control 
rho = 2.0           # dynamics penalty weight


# -----------------------------
# Initial condition
# -----------------------------
x0 = np.array([
    0.0,  0.0,   # p1
    3.0,  0.0,   # p2
    0.8,  0.0,   # v1
   -0.6,  0.0    # v2
])

x0 = np.array([
    -3.0,  0.0,   # p1
    3.0,  0.0,   # p2
    1.0,  1.0,   # v1
   -1.0,  1.0    # v2
])

d_encounter = 1.0
c1 = 10.0
c2 = 500.0

d_encounter = 2.5
c1 = 5000.0
c2 = 50000.0

sigma_deg = 30
tf_min = 2.0
tf_max = 15.0


# -----------------------------
# Params dictionary
# -----------------------------
params = {
    "N": N,
    "nx": nx,
    "nu": nu,
    "dt": dt,
    "tf": tf,
    "A": A,
    "B": B,
    "D": D,
    "Q1": Q1,
    "Q2": Q2,
    "alpha": alpha,
    "rho": rho,
    "x0": x0,
    "d_encounter": d_encounter,
    "c1": c1,
    "c2": c2,
    "sigma_deg": sigma_deg,
    "tf_min": tf_min,
    "tf_max": tf_max
}

# def unpack_z(z, params):
#     N = params["N"]
#     nx = params["nx"]
#     nu = params["nu"]

#     nX = N * nx

#     X_part = z[:nX]
#     U_part = z[nX:]

#     X_dec = X_part.reshape(N, nx)   # x1,...,x10
#     U = U_part.reshape(N, nu)       # u0,...,u9

#     return X_dec, U

def unpack_z(z, params):
    N = params["N"]
    nu = params["nu"]
    U = z.reshape(N, nu)
    return U


# def build_X_full(X_dec, params):
#     x0 = params["x0"]

#     N = X_dec.shape[0]
#     nx = X_dec.shape[1]

#     X_full = np.zeros((N + 1, nx))
#     X_full[0] = x0
#     X_full[1:] = X_dec

#     return X_full

def get_p1(x):
    return x[0:2]

def get_p2(x):
    return x[2:4]

def get_v1(x):
    return x[4:6]

def get_v2(x):
    return x[6:8]

def compute_r(x):
    p1 = get_p1(x)
    p2 = get_p2(x)
    return np.linalg.norm(p1 - p2)

def compute_theta(x):
    v1 = get_v1(x)
    v2 = get_v2(x)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-8 or norm_v2 < 1e-8:
        return 0.0

    # dot
    dot = np.dot(v1, v2)

    # 2D cross product (scalar)
    cross = v1[0]*v2[1] - v1[1]*v2[0]

    theta_rad = np.arctan2(cross, dot)

    # aralık: [-pi, pi] → [0, 2pi]
    if theta_rad < 0:
        theta_rad += 2 * np.pi

    return theta_rad

def compute_relative_angle_signed(x):
    p1 = get_p1(x)
    p2 = get_p2(x)
    v1 = get_v1(x)
    v2 = get_v2(x)

    p_rel = p2 - p1      # wheelchair'den obstacle'a konum vektörü
    v_rel = v2 - v1      # obstacle'ın wheelchair'e göre hızı

    norm_p = np.linalg.norm(p_rel)
    norm_v = np.linalg.norm(v_rel)

    if norm_p < 1e-8 or norm_v < 1e-8:
        return 0.0

    dot = np.dot(p_rel, v_rel)

    cross = p_rel[0]*v_rel[1] - p_rel[1]*v_rel[0]

    phi_rad = np.arctan2(cross, dot)

    # [-pi, pi] yerine [0, 2pi] istiyorsan:
    if phi_rad < 0:
        phi_rad += 2*np.pi

    return phi_rad

def f_of_r(r, params):
    d_encounter = params["d_encounter"]
    c1 = params["c1"]
    c2 = params["c2"]

    if r > d_encounter:
        return 0.0
    elif r > d_encounter / 1.5:
        return c1
    else:
        return c2

def compute_speeds(x):
    v1 = get_v1(x)   # wheelchair velocity
    v2 = get_v2(x)   # obstacle velocity

    v_w = np.linalg.norm(v1)
    v_obs = np.linalg.norm(v2)

    return v_w, v_obs

# def h_of_theta(theta_rad, v_w, v_obs, params):
#     """
#     theta_rad: angle between wheelchair velocity and obstacle velocity, in radians
#     v_w: wheelchair speed magnitude
#     v_obs: obstacle speed magnitude
#     sigma_deg: Gaussian width in degrees
#     """
#     sigma_deg = params["sigma_deg"]

#     theta_deg = np.degrees(theta_rad)

#     # numerical safety: put angle into [0, 360)
#     theta_deg = theta_deg % 360.0

#     # near-zero tolerance because exact 0 rarely happens numerically
#     tol = 1e-8

#     # theta = 0 or 360 case
#     if theta_deg <= tol or abs(theta_deg - 360.0) <= tol:
#         if v_w <= v_obs:
#             return 0.0
#         else:
#             # optimal control does not solve this case;
#             # later we handle it as a special rule if needed
#             return 0.0

#     # low-risk same-direction-ish region
#     if (0.0 < theta_deg <= 45.0) or (315.0 <= theta_deg < 360.0):
#         return 0.0

#     # Gaussian risk region
#     if 45.0 < theta_deg < 315.0:
#         return np.exp(-0.5 * ((theta_deg - 180.0) ** 2) / (sigma_deg ** 2))

#     return 0.0

def h_of_relative_angle(phi_rad, params):
    sigma_deg = params["sigma_deg"]

    phi_deg = np.degrees(phi_rad) % 360.0

    # 180 dereceye uzaklık
    diff = abs(phi_deg - 180.0)

    return np.exp(-0.5 * (diff**2) / (sigma_deg**2))

def reduce_speed_to_obstacle(x):
    v_w_vec = get_v1(x)
    v_obs_speed = np.linalg.norm(get_v2(x))
    v_w_speed = np.linalg.norm(v_w_vec)

    if v_w_speed < 1e-8:
        return v_w_vec.copy()

    return (v_obs_speed / v_w_speed) * v_w_vec #yön aynı kalır, sadece magnitude obstacle speed’e iner.

def should_run_oc(x, params, tol=1e-6):
    r = compute_r(x)
    f_val = f_of_r(r, params)

    # 1) Distance gate
    if f_val == 0.0:
        return False, "no_interaction", None

    theta_rad = compute_theta(x)
    theta_deg = np.degrees(theta_rad) % 360.0
    v_w, v_obs = compute_speeds(x)

    phi_rad = compute_relative_angle_signed(x)

    # 2) Same-direction case
    if theta_deg < tol or abs(theta_deg - 360.0) < tol:
        if v_w > v_obs:
            v1_new = reduce_speed_to_obstacle(x)
            return False, "reduce_speed", v1_new #onunden giden adamin hizina indirsen yeter.
        else:
            return False, "safe", None

    # 3) General collision potential
    #h_val = h_of_theta(theta_rad, v_w, v_obs, params)
    h_val = h_of_relative_angle(phi_rad, params)
    P_collision = f_val * h_val

    if P_collision > tol:
        return True, "run_oc", None
    else:
        return False, "safe", None
    
def collision_potential(x, params):
    r = compute_r(x)
    #theta_rad = compute_theta(x)
    #v_w, v_obs = compute_speeds(x)
    #h_val = h_of_theta(theta_rad, v_w, v_obs, params)
    
    phi = compute_relative_angle_signed(x)
    f_val = f_of_r(r, params)
    h_val = h_of_relative_angle(phi, params)

    return f_val * h_val

# def compute_Jd(X_full, U, params):
#     N = params["N"]
#     dt = params["dt"]
#     D = params["D"]
#     alpha = params["alpha"]

#     J_state = 0.0
#     J_control = 0.0

#     for i in range(N):
#         xi = X_full[i]
#         xip1 = X_full[i + 1]
#         ui = U[i]

#         # node i
#         ri = compute_r(xi)
#         theta_i = compute_theta(xi)
#         v_w_i, v_obs_i = compute_speeds(xi)

#         L_i = (
#             f_of_r(ri, params)
#             + h_of_theta(theta_i, v_w_i, v_obs_i, params)
#         )

#         # node i+1
#         rip1 = compute_r(xip1)
#         theta_ip1 = compute_theta(xip1)
#         v_w_ip1, v_obs_ip1 = compute_speeds(xip1)

#         L_ip1 = (
#             f_of_r(rip1, params)
#             + h_of_theta(theta_ip1, v_w_ip1, v_obs_ip1, params)
#         )

#         # trapezoidal state/risk cost
#         J_state += (dt / 2.0) * (L_i + L_ip1)

#         # interval control cost
#         J_control += dt * alpha * (ui.T @ D @ ui)

#     return J_state + J_control

def safe_normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def cross_track_error(p, p0, v0):
    """
    p'nin, p0 + s*v0 çizgisine dik uzaklığını verir.
    Burada s serbest; yani hız büyüklüğünü/timing'i cezalandırmaz.
    """
    d = safe_normalize(v0)

    if np.linalg.norm(d) < 1e-8:
        return 0.0

    # 2D'de path direction'a dik normal
    n = np.array([-d[1], d[0]])

    return np.dot(p - p0, n)

def compute_Jd(X_full, U, params):
    N = params["N"]
    dt = params["dt"]
    D = params["D"]
    alpha = params["alpha"]

    q_dir = params.get("q_dir", 500.0)
    q_line = params.get("q_line", 500.0)
    q_terminal_dir = params.get("q_terminal_dir", 5000.0)
    q_terminal_line = params.get("q_terminal_line", 5000.0)
    k_recovery = params.get("k_recovery", 5.0)

    x_initial = X_full[0]

    p1_initial = get_p1(x_initial)
    p2_initial = get_p2(x_initial)

    v1_initial = get_v1(x_initial)
    v2_initial = get_v2(x_initial)

    v1_initial_dir = safe_normalize(v1_initial)
    v2_initial_dir = safe_normalize(v2_initial)

    J_risk = 0.0
    J_control = 0.0
    J_direction = 0.0
    J_line = 0.0

    for i in range(N):
        xi = X_full[i]
        xip1 = X_full[i + 1]
        ui = U[i]

        # -----------------------------
        # risk cost
        # -----------------------------
        ri = compute_r(xi)
        rip1 = compute_r(xip1)

        phi_i = compute_relative_angle_signed(xi)
        phi_ip1 = compute_relative_angle_signed(xip1)

        L_i = f_of_r(ri, params) * h_of_relative_angle(phi_i, params)
        L_ip1 = f_of_r(rip1, params) * h_of_relative_angle(phi_ip1, params)

        J_risk += (dt / 2.0) * (L_i + L_ip1)

        # -----------------------------
        # recovery weight
        # Zaman ilerledikçe başlangıç yönüne/path'ine dönüş daha önemli olsun.
        # -----------------------------
        w_i = 1.0 + k_recovery * (i / N)**2
        w_ip1 = 1.0 + k_recovery * ((i + 1) / N)**2

        # -----------------------------
        # velocity direction cost
        # magnitude önemli değil, sadece yön önemli.
        # -----------------------------
        v1_dir_i = safe_normalize(get_v1(xi))
        v2_dir_i = safe_normalize(get_v2(xi))

        v1_dir_ip1 = safe_normalize(get_v1(xip1))
        v2_dir_ip1 = safe_normalize(get_v2(xip1))

        dv1_dir_i = v1_dir_i - v1_initial_dir
        dv2_dir_i = v2_dir_i - v2_initial_dir

        dv1_dir_ip1 = v1_dir_ip1 - v1_initial_dir
        dv2_dir_ip1 = v2_dir_ip1 - v2_initial_dir

        Dir_i = q_dir * (dv1_dir_i @ dv1_dir_i + dv2_dir_i @ dv2_dir_i)
        Dir_ip1 = q_dir * (dv1_dir_ip1 @ dv1_dir_ip1 + dv2_dir_ip1 @ dv2_dir_ip1)

        J_direction += (dt / 2.0) * (w_i * Dir_i + w_ip1 * Dir_ip1)

        # -----------------------------
        # line recovery cost
        # Başlangıç trajectory çizgisine dik uzaklık.
        # Bu, hız büyüklüğünü/timing'i cezalandırmaz.
        # -----------------------------
        e1_i = cross_track_error(get_p1(xi), p1_initial, v1_initial)
        e2_i = cross_track_error(get_p2(xi), p2_initial, v2_initial)

        e1_ip1 = cross_track_error(get_p1(xip1), p1_initial, v1_initial)
        e2_ip1 = cross_track_error(get_p2(xip1), p2_initial, v2_initial)

        Line_i = q_line * (e1_i**2 + e2_i**2)
        Line_ip1 = q_line * (e1_ip1**2 + e2_ip1**2)

        J_line += (dt / 2.0) * (w_i * Line_i + w_ip1 * Line_ip1)

        # -----------------------------
        # control effort
        # -----------------------------
        J_control += dt * alpha * (ui.T @ D @ ui)

    # -----------------------------
    # terminal direction + terminal line recovery
    # En sonda mutlaka başlangıç yönüne/path çizgisine dönsün.
    # -----------------------------
    xf = X_full[-1]

    v1f_dir = safe_normalize(get_v1(xf))
    v2f_dir = safe_normalize(get_v2(xf))

    dv1f_dir = v1f_dir - v1_initial_dir
    dv2f_dir = v2f_dir - v2_initial_dir

    J_terminal_dir = q_terminal_dir * (
        dv1f_dir @ dv1f_dir + dv2f_dir @ dv2f_dir
    )

    e1f = cross_track_error(get_p1(xf), p1_initial, v1_initial)
    e2f = cross_track_error(get_p2(xf), p2_initial, v2_initial)

    J_terminal_line = q_terminal_line * (e1f**2 + e2f**2)

    return (
        J_risk
        + J_direction
        + J_line
        + J_terminal_dir
        + J_terminal_line
        + J_control
    )

def compute_Jd_components(X_full, U, params):
    N = params["N"]
    dt = params["dt"]
    D = params["D"]
    alpha = params["alpha"]

    q_dir = params.get("q_dir", 500.0)
    q_line = params.get("q_line", 500.0)
    q_terminal_dir = params.get("q_terminal_dir", 5000.0)
    q_terminal_line = params.get("q_terminal_line", 5000.0)
    k_recovery = params.get("k_recovery", 5.0)

    x_initial = X_full[0]

    p1_initial = get_p1(x_initial)
    p2_initial = get_p2(x_initial)

    v1_initial = get_v1(x_initial)
    v2_initial = get_v2(x_initial)

    v1_initial_dir = safe_normalize(v1_initial)
    v2_initial_dir = safe_normalize(v2_initial)

    J_risk = 0.0
    J_control = 0.0
    J_direction = 0.0
    J_line = 0.0

    for i in range(N):
        xi = X_full[i]
        xip1 = X_full[i + 1]
        ui = U[i]

        # -----------------------------
        # risk cost
        # -----------------------------
        ri = compute_r(xi)
        rip1 = compute_r(xip1)

        phi_i = compute_relative_angle_signed(xi)
        phi_ip1 = compute_relative_angle_signed(xip1)

        L_i = f_of_r(ri, params) * h_of_relative_angle(phi_i, params)
        L_ip1 = f_of_r(rip1, params) * h_of_relative_angle(phi_ip1, params)

        J_risk += (dt / 2.0) * (L_i + L_ip1)

        # -----------------------------
        # recovery weight
        # Zaman ilerledikçe başlangıç yönüne/path'ine dönüş daha önemli olsun.
        # -----------------------------
        w_i = 1.0 + k_recovery * (i / N)**2
        w_ip1 = 1.0 + k_recovery * ((i + 1) / N)**2

        # -----------------------------
        # velocity direction cost
        # magnitude önemli değil, sadece yön önemli.
        # -----------------------------
        v1_dir_i = safe_normalize(get_v1(xi))
        v2_dir_i = safe_normalize(get_v2(xi))

        v1_dir_ip1 = safe_normalize(get_v1(xip1))
        v2_dir_ip1 = safe_normalize(get_v2(xip1))

        dv1_dir_i = v1_dir_i - v1_initial_dir
        dv2_dir_i = v2_dir_i - v2_initial_dir

        dv1_dir_ip1 = v1_dir_ip1 - v1_initial_dir
        dv2_dir_ip1 = v2_dir_ip1 - v2_initial_dir

        Dir_i = q_dir * (dv1_dir_i @ dv1_dir_i + dv2_dir_i @ dv2_dir_i)
        Dir_ip1 = q_dir * (dv1_dir_ip1 @ dv1_dir_ip1 + dv2_dir_ip1 @ dv2_dir_ip1)

        J_direction += (dt / 2.0) * (w_i * Dir_i + w_ip1 * Dir_ip1)

        # -----------------------------
        # line recovery cost
        # Başlangıç trajectory çizgisine dik uzaklık.
        # Bu, hız büyüklüğünü/timing'i cezalandırmaz.
        # -----------------------------
        e1_i = cross_track_error(get_p1(xi), p1_initial, v1_initial)
        e2_i = cross_track_error(get_p2(xi), p2_initial, v2_initial)

        e1_ip1 = cross_track_error(get_p1(xip1), p1_initial, v1_initial)
        e2_ip1 = cross_track_error(get_p2(xip1), p2_initial, v2_initial)

        Line_i = q_line * (e1_i**2 + e2_i**2)
        Line_ip1 = q_line * (e1_ip1**2 + e2_ip1**2)

        J_line += (dt / 2.0) * (w_i * Line_i + w_ip1 * Line_ip1)

        # -----------------------------
        # control effort
        # -----------------------------
        J_control += dt * alpha * (ui.T @ D @ ui)

    # -----------------------------
    # terminal direction + terminal line recovery
    # En sonda mutlaka başlangıç yönüne/path çizgisine dönsün.
    # -----------------------------
    xf = X_full[-1]

    v1f_dir = safe_normalize(get_v1(xf))
    v2f_dir = safe_normalize(get_v2(xf))

    dv1f_dir = v1f_dir - v1_initial_dir
    dv2f_dir = v2f_dir - v2_initial_dir

    J_terminal_dir = q_terminal_dir * (
        dv1f_dir @ dv1f_dir + dv2f_dir @ dv2f_dir
    )

    e1f = cross_track_error(get_p1(xf), p1_initial, v1_initial)
    e2f = cross_track_error(get_p2(xf), p2_initial, v2_initial)

    J_terminal_line = q_terminal_line * (e1f**2 + e2f**2)

    Jd  = J_risk + J_direction + J_line + J_terminal_dir + J_terminal_line + J_control

    print("=" * 45)
    print(f"  J_risk          = {J_risk:>12.4f}")
    print(f"  J_direction     = {J_direction:>12.4f}")
    print(f"  J_line          = {J_line:>12.4f}")
    print(f"  J_terminal_dir  = {J_terminal_dir:>12.4f}")
    print(f"  J_terminal_line = {J_terminal_line:>12.4f}")
    print(f"  J_control       = {J_control:>12.4f}")
    print("=" * 45)

    return(Jd)


# def compute_Jd(X_full, U, params):
#     #beta = 50.0
#     #J_smooth = 0.0

#     N = params["N"]
#     dt = params["dt"]
#     D = params["D"]
#     Q1 = params["Q1"]   # 2x2
#     Q2 = params["Q2"]   # 2x2
#     alpha = params["alpha"]

#     x_initial = X_full[0]
#     v1_initial = get_v1(x_initial)
#     v2_initial = get_v2(x_initial)

#     v1_initial_dir = safe_normalize(v1_initial)
#     v2_initial_dir = safe_normalize(v2_initial)

#     J_risk = 0.0
#     J_control = 0.0
#     J_velocity = 0.0

#     for i in range(N):
#         xi = X_full[i]
#         xip1 = X_full[i + 1]
#         ui = U[i]

#         # -----------------------------
#         # risk cost at node i
#         # -----------------------------
#         ri = compute_r(xi)
#         #theta_i = compute_theta(xi)
#         #v_w_i, v_obs_i = compute_speeds(xi)

#         # L_i = (
#         #     f_of_r(ri, params) * h_of_theta(theta_i, v_w_i, v_obs_i, params)
#         # )

#         phi_i = compute_relative_angle_signed(xi)
#         L_i = f_of_r(ri, params) * h_of_relative_angle(phi_i, params)

#         # -----------------------------
#         # risk cost at node i+1
#         # -----------------------------
#         rip1 = compute_r(xip1)
#         #theta_ip1 = compute_theta(xip1)
#         #v_w_ip1, v_obs_ip1 = compute_speeds(xip1)

#         # L_ip1 = (
#         #     f_of_r(rip1, params) * h_of_theta(theta_ip1, v_w_ip1, v_obs_ip1, params)
#         # )

#         phi_ip1 = compute_relative_angle_signed(xip1)
#         L_ip1 = f_of_r(rip1, params) * h_of_relative_angle(phi_ip1, params)

#         J_risk += (dt / 2.0) * (L_i + L_ip1)

#         # -----------------------------
#         # velocity deviation cost: integralin icine Jterm. Her andaki hizlar baslangic hizina benzesin.
#         # -----------------------------
#         # dv1_i = get_v1(xi) - v1_initial
#         # dv2_i = get_v2(xi) - v2_initial

#         # dv1_ip1 = get_v1(xip1) - v1_initial
#         # dv2_ip1 = get_v2(xip1) - v2_initial

#         dv1_i = safe_normalize(get_v1(xi)) - v1_initial_dir
#         dv2_i = safe_normalize(get_v2(xi)) - v2_initial_dir

#         dv1_ip1 = safe_normalize(get_v1(xip1)) - v1_initial_dir
#         dv2_ip1 = safe_normalize(get_v2(xip1)) - v2_initial_dir

#         V_i = dv1_i.T @ Q1 @ dv1_i + dv2_i.T @ Q2 @ dv2_i
#         V_ip1 = dv1_ip1.T @ Q1 @ dv1_ip1 + dv2_ip1.T @ Q2 @ dv2_ip1

#         J_velocity += (dt / 2.0) * (V_i + V_ip1)

#         # -----------------------------
#         # control effort
#         # -----------------------------
#         J_control += dt * alpha * (ui.T @ D @ ui)

#         # smoothness of control
#         # if i < N - 1:
#         #     du = U[i + 1] - U[i]
#         #     J_smooth += beta * np.sum(du**2)


#     return J_risk + J_velocity + J_control 

# def compute_Jterm(X_full, params):
#     #x=[p1x​,p1y​,p2x​,p2y​,v1x​,v1y​,v2x​,v2y​]T
    
#     Q1 = params["Q1"]
#     Q2 = params["Q2"]

#     x_initial = X_full[0]
#     x_final = X_full[-1]

#     dx1 = np.concatenate([
#         get_p1(x_final) - get_p1(x_initial),
#         get_v1(x_final) - get_v1(x_initial)
#     ])

#     dx2 = np.concatenate([
#         get_p2(x_final) - get_p2(x_initial),
#         get_v2(x_final) - get_v2(x_initial)
#     ])

#     Jterm1 = dx1.T @ Q1 @ dx1
#     Jterm2 = dx2.T @ Q2 @ dx2

#     return Jterm1 + Jterm2

# def compute_Jterm(X_full, params):
#     #x=[p1x​,p1y​,p2x​,p2y​,v1x​,v1y​,v2x​,v2y​]T
    
#     Q1 = params["Q1"]
#     Q2 = params["Q2"]

#     x_initial = X_full[0]
#     x_final = X_full[-1]

#     dx1 = get_v1(x_final) - get_v1(x_initial)  # shape: (2,)
#     dx2 = get_v2(x_final) - get_v2(x_initial)  # shape: (2,)

#     Jterm1 = dx1.T @ Q1 @ dx1
#     Jterm2 = dx2.T @ Q2 @ dx2

#     return Jterm1 + Jterm2

# def compute_Jpen(X_full, U, params):
#     N = params["N"]
#     dt = params["dt"]
#     A = params["A"]
#     B = params["B"]
#     rho = params["rho"]

#     Jpen = 0.0

#     # e0: forward difference
#     x0 = X_full[0]
#     x1 = X_full[1]
#     u0 = U[0]

#     e0 = (x1 - x0) / dt - (A @ x0 + B @ u0)

#     Jpen += rho * (e0.T @ e0)

#     for i in range(1, N):
#         xim1 = X_full[i - 1]
#         xi   = X_full[i]
#         xip1 = X_full[i + 1]
#         ui   = U[i]

#         ei = (xip1 - xim1) / (2.0 * dt) - (A @ xi + B @ ui)

#         Jpen += rho * (ei.T @ ei)
    
#     return Jpen

# def objective(z, params):
#     X_dec, U = unpack_z(z, params)
#     X_full = build_X_full(X_dec, params)

#     Jd = compute_Jd(X_full, U, params)
#     Jterm = compute_Jterm(X_full, params)
#     Jpen = compute_Jpen(X_full, U, params)

#     return Jd + Jterm + Jpen

def compute_tf(x, params):
    tf_min = params["tf_min"]
    tf_max = params["tf_max"]

    p1 = get_p1(x)
    p2 = get_p2(x)

    v1 = get_v1(x)
    v2 = get_v2(x)

    r_vec = p2 - p1
    dist = np.linalg.norm(r_vec)

    if dist < 1e-6:
        return tf_min

    r_hat = r_vec / dist #wheelchair’den obstacle’a doğru yön (unit vector)
    v_rel = v2 - v1      #obstacle sana göre nasıl hareket ediyor?

    v_close = -np.dot(v_rel, r_hat) #relative velocity’nin wheelchair–obstacle doğrultusundaki bileşeni

    if v_close <= 1e-6:
        tf = tf_max
    else:
        tf = 2 * dist / v_close #bu iki ajan ne kadar sürede birbirine yaklaşır?

    # clamp
    tf = np.clip(tf, tf_min, tf_max)

    return tf

def pso_objective(Z, params):
    n_particles = Z.shape[0]
    costs = np.zeros(n_particles)
    for i in range(n_particles):
        costs[i] = objective(Z[i], params)
    return costs

# def rollout_dynamics_central(U, params):
#     N = params["N"]
#     nx = params["nx"]
#     dt = params["dt"]
#     A = params["A"]
#     B = params["B"]

#     X_full = np.zeros((N + 1, nx))
#     X_full[0] = params["x0"]

#     # İlk adım forward difference olmak zorunda
#     X_full[1] = X_full[0] + dt * (A @ X_full[0] + B @ U[0])

#     # Geri kalanlar central difference
#     for i in range(1, N):
#         X_full[i + 1] = (
#             X_full[i - 1]
#             + 2.0 * dt * (A @ X_full[i] + B @ U[i])
#         )

#     return X_full

def rollout_dynamics(U, params):
    N  = params["N"]
    nx = params["nx"]
    dt = params["dt"]
    A  = params["A"]
    B  = params["B"]

    X_full = np.zeros((N + 1, nx))
    X_full[0] = params["x0"]

    for i in range(N):
        X_full[i + 1] = X_full[i] + dt * (A @ X_full[i] + B @ U[i])

    return X_full

# def rollout_dynamics(U, params):
#     N = params["N"]
#     nx = params["nx"]
#     dt = params["dt"]

#     X_full = np.zeros((N + 1, nx))
#     X_full[0] = params["x0"]

#     for i in range(N):
#         x = X_full[i]

#         p1 = x[0:2]
#         p2 = x[2:4]
#         v1 = x[4:6]
#         v2 = x[6:8]

#         u1 = U[i, 0:2]
#         u2 = U[i, 2:4]

#         X_full[i + 1, 0:2] = p1 + dt * v1 + 0.5 * dt**2 * u1
#         X_full[i + 1, 2:4] = p2 + dt * v2 + 0.5 * dt**2 * u2

#         X_full[i + 1, 4:6] = v1 + dt * u1
#         X_full[i + 1, 6:8] = v2 + dt * u2

#     return X_full


def objective(z, params):
    U = unpack_z(z, params)
    X_full = rollout_dynamics(U, params)

    Jd = compute_Jd(X_full, U, params)
    #Jterm = compute_Jterm(X_full, params)

    return Jd


#control vektoru cok buyuk olamaz, sinirlidir. Velocity'e sinir koy, unbounded/infinite acceleration olamaz, sonsuz gucle zink diye hizini degistiriyor demek, fiziksel sistemlerde moment of intertia var.
#5 ise wheelchair icin, onun carpacagi insanlar icin 10 diyebilirsin. 
#Optimizasyon sagladiktan sonra bound lari sagliyor mu diye check edersin. 
#bundan sonra PSO run edeceksin.
#once basit, cozumunu bildigin bir problemde dene, sonra senin probleme uygula. 
#optimum state, state denklemlerden bulacaksin.
#optimizasyon ne verdi? karsilastir ona gore penalty katsayisini tune et.
#state aslinda dependent degisken optimizasyon basitlestirmek icin independent degisken gibi aldik onu da. 


if __name__ == "__main__":

    # -----------------------------
    # 1) tf hesapla
    # -----------------------------
    #tf = compute_tf(params["x0"], params)
    tf = 4.5

    params["tf"] = tf
    params["dt"] = tf / params["N"]

    print("tf =", tf)
    print("dt =", params["dt"])

    # -----------------------------
    # 2) Bounds (direkt burada)
    # -----------------------------

    # pos_bound  = 10.0   # pozisyon ±10 m
    # vel_bound  = 2.0    # hız ±2 m/s
    ctrl_bound = 100.0    # kontrol (ivme) ±5 m/s²

    # x_min = np.array([
    #     -pos_bound, -pos_bound,
    #     -pos_bound, -pos_bound,
    #     -vel_bound, -vel_bound,
    #     -vel_bound, -vel_bound
    # ])

    # x_max = np.array([
    #     pos_bound, pos_bound,
    #     pos_bound, pos_bound,
    #     vel_bound, vel_bound,
    #     vel_bound, vel_bound
    # ])

    u_min = np.array([
        -ctrl_bound, -ctrl_bound,
        -ctrl_bound, -ctrl_bound
    ])

    u_max = np.array([
        ctrl_bound, ctrl_bound,
        ctrl_bound, ctrl_bound
    ])

    lower_bounds = np.tile(u_min, N)
    upper_bounds = np.tile(u_max, N)

    bounds = (lower_bounds, upper_bounds)

    # -----------------------------
    # 3) PSO ayarları
    # -----------------------------
    options = {
        "c1": 1.5,
        "c2": 1.5,
        "w": 0.7
    }

    optimizer = ps.single.GlobalBestPSO(
        n_particles=50, #200
        dimensions=n_dec,
        options=options,
        bounds=bounds
    )

    # -----------------------------
    # 4) Run
    # -----------------------------
    
    #Jd(running cost: carpisma riski + control effort)
    #Jterm(baslangictan ne kadar saptin)
    #Jpen(dynamics ne kadar ihlal edildi)


    best_cost, best_z = optimizer.optimize(
        pso_objective,
        iters=2000,
        params=params
    )

    print("Best cost:", best_cost)

    # -----------------------------
    # 5) Aç
    # -----------------------------
    U_best = unpack_z(best_z, params)
    X_full_best = rollout_dynamics(U_best, params)

    anim = animate_trajectory(X_full_best, params, interval=400)

    print("=" * 40)
    print(f"En iyi maliyet : {best_cost:.4f}")
    print(f"Jd   = {compute_Jd_components(X_full_best, U_best, params):.4f}")
    #print(f"Jd   = {compute_Jd(X_full_best, U_best, params):.4f}")
    #print(f"Jterm= {compute_Jterm(X_full_best, params):.4f}")
    #print(f"Jpen = {compute_Jpen(X_full_best, U_best, params):.4f}")
    print("=" * 40)
    print(f"p1 başlangıç : {X_full_best[0, 0:2]}")
    print(f"p1 bitiş     : {X_full_best[-1, 0:2]}")
    print(f"p2 başlangıç : {X_full_best[0, 2:4]}")
    print(f"p2 bitiş     : {X_full_best[-1, 2:4]}")
    print(f"v1 başlangıç : {X_full_best[0, 4:6]}")
    print(f"v1 bitiş     : {X_full_best[-1, 4:6]}")
    print(f"v2 başlangıç : {X_full_best[0, 6:8]}")
    print(f"v2 bitiş     : {X_full_best[-1, 6:8]}")


    # ── Plot 1: Maliyet geçmişi ──────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(optimizer.cost_history)
    plt.xlabel("İterasyon")
    plt.ylabel("En iyi maliyet")
    plt.title("PSO — Maliyet geçmişi")
    plt.yscale("log")   # log scale: başta büyük, sonda küçük değerleri iyi gösterir
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── Plot 2: Trajektori (p1 ve p2 yolu) ───────────────────
    p1_traj = X_full_best[:, 0:2]   # (11, 2)
    p2_traj = X_full_best[:, 2:4]   # (11, 2)

    plt.figure(figsize=(7, 7))
    plt.plot(p1_traj[:, 0], p1_traj[:, 1], 'b-o', label='p1 (wheelchair)')
    plt.plot(p2_traj[:, 0], p2_traj[:, 1], 'r-o', label='p2 (obstacle)')
    plt.plot(p1_traj[0, 0],  p1_traj[0, 1],  'bs', markersize=10, label='p1 start')
    plt.plot(p2_traj[0, 0],  p2_traj[0, 1],  'rs', markersize=10, label='p2 start')
    plt.plot(p1_traj[-1, 0], p1_traj[-1, 1], 'b*', markersize=14, label='p1 end')
    plt.plot(p2_traj[-1, 0], p2_traj[-1, 1], 'r*', markersize=14, label='p2 end')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajektori — p1 ve p2")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── Plot 3: Hiz (p1 ve p2) ───────────────────
    v1_vel = np.zeros(X_full_best.shape[0])
    v2_vel = np.zeros(X_full_best.shape[0])

    for i in range(X_full_best.shape[0]):
        v1_vel[i] = np.linalg.norm(X_full_best[i, 4:6])
        v2_vel[i] = np.linalg.norm(X_full_best[i, 6:8])

    plt.figure(figsize=(10, 4))
    plt.plot(v1_vel, 'b-o', label='p1 (wheelchair)')
    plt.plot(v2_vel, 'r-o', label='p2 (obstacle)')
    plt.xlabel("time step")
    plt.ylabel("hız (m/s)")
    plt.title("Hız — v1 ve v2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#NOTLAR:
#Penalty terimine gerek yok. Otomatik olarak sagliyoruz. State'leri equality constraint'den hesapliyoruz.
#State'leri optimizasyonda bilinmeyenden cikar.
#Basa donduruyor, zig-zag yapiyor o yuzden. Carpisma durumundan cikarken baslangic konumlari esit olacak dedik.
#Istedigimiz baslangictaki davranisa benzer bir sekilde hareket etmek. Sadece hizlar. 

    r_vals = np.linalg.norm(
    X_full_best[:, 0:2] - X_full_best[:, 2:4],
    axis=1)

    collision_vals = np.array([
        collision_potential(X_full_best[i], params)
        for i in range(X_full_best.shape[0])
    ])

    plt.figure()
    plt.plot(r_vals, 'o-', label='distance r')
    plt.axhline(params["d_encounter"], linestyle='--', label='d_encounter')
    plt.xlabel("time step")
    plt.ylabel("distance")
    plt.title("p1-p2 distance over time")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(collision_vals, 'o-', label='collision potential')
    plt.xlabel("time step")
    plt.ylabel("f(r) * h(phi)")
    plt.title("Collision potential over time")
    plt.grid(True)
    plt.legend()
    plt.show()

    #we should plot the h(theta) over the trajectory
    # vw_vals   = np.zeros(X_full_best.shape[0])
    # vobs_vals = np.zeros(X_full_best.shape[0])
    # theta     = np.zeros(X_full_best.shape[0])

    # for i in range(X_full_best.shape[0]): 
    #     vw_vals[i]   = compute_speeds(X_full_best[i])[0]
    #     vobs_vals[i] = compute_speeds(X_full_best[i])[1]
    #     theta[i]     = compute_theta(X_full_best[i])

    # h_of_theta_vals = np.array([
    # h_of_theta(theta[i], vw_vals [i], vobs_vals[i], params)
    # for i in range(X_full_best.shape[0])
    # ])

    # plt.figure()
    # plt.plot(h_of_theta_vals, 'o-', label='h(theta) values over the trajectory')
    # plt.xlabel("time step")
    # plt.ylabel("h(theta)")
    # plt.title("h(theta) over time")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    phi = np.zeros(X_full_best.shape[0])

    for i in range(X_full_best.shape[0]): 
        phi[i] = compute_relative_angle_signed(X_full_best[i])

    h_of_phi_vals = np.array([
    h_of_relative_angle(phi[i], params)
    for i in range(X_full_best.shape[0])
    ])

    plt.figure()
    plt.plot(h_of_phi_vals, 'o-', label='h(phi) values over the trajectory')
    plt.xlabel("time step")
    plt.ylabel("h(phi)")
    plt.title("h(phi) over time")
    plt.grid(True)
    plt.legend()
    plt.show()