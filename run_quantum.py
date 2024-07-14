import quantum_functions as qf

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

epsilon = 10**(-17)
print("About He:")
coeff_1s = [1.347900, -0.001613, -0.100506, -0.270779, 0.0, 0.0, 0.0]
coeff_2s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
coeff_2p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Define symbolic variables
r = sp.symbols("r")
k = sp.symbols("k")
s_R = []
s_K = []
s_tot = []
zs = range(2, 11)
z_s = [1.4595, 5.3244, 2.6298, 1.7504, 1, 1, 1]
z_p = [1, 1, 1, 1, 1, 1, 1]

R1s, R2s, R2p = qf.R_one_1s(coeff_1s, coeff_2s, coeff_2p, z_s, z_p)

t = []

p_R = 1 / (4 * np.pi * 2) * (2 * R1s**2 + 0.0 * R2s**2 + 0.0 * R2p**2)



qf.plot_radius(p_R, 1.3, "DOS of He in the position space", "r", "rho(r)", "He")


S_r = qf.integrate_radius_to_inf(-4 * sp.pi * p_R * sp.log(p_R + epsilon) * r**2)
s_R.append(S_r)


K1s, K2s, K2p = qf.k_one_1s(coeff_1s, coeff_2s, coeff_2p, z_s, z_p)


n_k = 1 / (4 * np.pi * 2) * (2 * K1s**2 + 0.0 * K2s**2 + 0.0 * K2p**2)


t.append(['S_r', S_r])

qf.plot_k(n_k, 3, "DOS of He in the k space", "k", "n(k)", "He")

S_k = qf.integrate_k_to_inf(-4 * sp.pi * n_k * sp.log(n_k + epsilon) * k**2)
s_K.append(S_k)
t.append(['S_k', S_k])

S_tot = S_r + S_k
s_tot.append(S_tot)
t.append(['S_tot', S_tot])

df = pd.DataFrame(t, columns=["Quantity", "Result"])
pd.options.display.float_format = '{:.12g}'.format
print(df)
print("\n")

# Repeat the above for elements 2 to 9 with their respective coefficients and z values.
elements = [
    {"name": "Li", "Z": 3, "coeff_1s": [0.141279, 0.874231, -0.005201, -0.002307, 0.006985, -0.000305, 0.000760],
     "coeff_2s": [-0.022416, -0.135791, 0.000389, -0.000068, -0.076544, 0.340542, 0.715708], "coeff_2p": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     "z_s": [4.3069, 2.4573, 6.7850, 7.4527, 1.8504, 0.7667, 0.6364], "z_p": [1, 1, 1, 1, 1, 1, 1],"a":2,"b":1,"c":0},
    
    {"name": "Be", "Z": 4, "coeff_1s": [0.285107, 0.474813, -0.001620, 0.052852, 0.243499, 0.000106, -0.000032],
     "coeff_2s": [-0.016378, -0.155066, 0.000426, -0.059234, -0.031925, 0.387968, 0.685674], "coeff_2p": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     "z_s": [5.7531, 3.7156, 9.9670, 3.7128, 4.4661, 1.2919, 0.8555], "z_p": [1, 1, 1, 1, 1, 1, 1],"a":2,"b":2,"c":0},

    {"name": "B", "Z": 5, "coeff_1s": [0.381607, 0.423958, -0.001316, -0.000822, 0.237016, 0.001062, -0.000137],
     "coeff_2s": [-0.022549, 0.321716, -0.000452, -0.072032, -0.050313, -0.484281, -0.518986], "coeff_2p": [0.007600, 0.045137, 0.184206, 0.394754, 0.432795, 0.0, 0.0],
     "z_s": [7.0178, 3.9468, 12.7297, 2.7646, 5.7420, 1.5436, 1.0802], "z_p": [5.7416, 2.6341, 1.8340, 1.1919, 0.8494, 1, 1],"a":2,"b":2,"c":1},

    {"name": "C", "Z": 6, "coeff_1s": [0.352872, 0.473621, -0.001199, 0.210887, 0.000886, 0.000465, -0.000119],
     "coeff_2s": [-0.071727, 0.438307, -0.000383, -0.091194, -0.393105, -0.579121, -0.126067], "coeff_2p": [0.006977, 0.070877, 0.230802, 0.411931, 0.350701, 0.0, 0.0],
     "z_s": [8.4936, 4.8788, 15.4660, 7.0500, 2.2640, 1.4747, 1.1639], "z_p": [7.0500, 3.2275, 2.1908, 1.4413, 1.0242, 1, 1],"a":2,"b":2,"c":2},

    {"name": "N", "Z": 7, "coeff_1s": [0.354839, 0.472579, -0.001038, 0.208492, 0.001687, 0.000206, 0.000064],
     "coeff_2s": [-0.067498, 0.434142, -0.000315, -0.080331, -0.374128, -0.522775, -0.207735], "coeff_2p": [0.006323, 0.082938, 0.260147, 0.418361, 0.308272, 0.0, 0.0],
     "z_s": [9.9051, 5.7429, 17.9816, 8.3087, 2.7611, 1.8223, 1.4191], "z_p": [8.3490, 3.8827, 2.5920, 1.6946, 1.1914, 1, 1],"a":2,"b":2,"c":3},

    {"name": "O", "Z": 8, "coeff_1s": [0.360063, 0.466625, -0.000918, 0.208441, 0.002018, 0.000216, 0.000133],
     "coeff_2s": [-0.064363, 0.433186, -0.000275, -0.072497, -0.369900, -0.512627, -0.227421], "coeff_2p": [0.005626, 0.126618, 0.328966, 0.395422, 0.231788, 0.0, 0.0],
     "z_s": [11.2970, 6.5966, 20.5019, 9.5546, 3.2482, 2.1608, 1.6411], "z_p": [9.6471, 4.3323, 2.7502, 1.7525, 1.2473, 1, 1],"a":2,"b":2,"c":4},

    {"name": "F", "Z": 9, "coeff_1s": [0.377498, 0.443947, -0.000797, 0.213846, 0.002183, 0.000335, 0.000147],
     "coeff_2s": [-0.058489, 0.426450, -0.000274, -0.063457, -0.358939, -0.516660, -0.239143], "coeff_2p": [0.004879, 0.130794, 0.337876, 0.396122, 0.225374, 0.0, 0.0],
     "z_s": [12.6074, 7.4101, 23.2475, 10.7416, 3.7543, 2.5009, 1.8577], "z_p": [11.0134, 4.9962, 3.1540, 1.9722, 1.3632, 1, 1],"a":2,"b":2,"c":5},

    {"name": "Ne", "Z": 10, "coeff_1s": [0.392290, 0.425817, -0.000702, 0.217206, 0.002300, 0.000463, 0.000147],
     "coeff_2s": [-0.053023, 0.419502, -0.000263, -0.055723, -0.349457, -0.523070, -0.246038], "coeff_2p": [0.004391, 0.133955, 0.342978, 0.395742, 0.221831, 0.0, 0.0],
     "z_s": [13.9074, 8.2187, 26.0325, 11.9249, 4.2635, 2.8357, 2.0715], "z_p": [12.3239, 5.6525, 3.5570, 2.2056, 1.4948, 1, 1],"a":2,"b":2,"c":6}
]

for element in elements:
    print(f"About {element['name']}:")
    if element["name"] in ["Be", "B"]:
        R1s, R2s, R2p = qf.R_two_3s(element["coeff_1s"], element["coeff_2s"], element["coeff_2p"], element["z_s"], element["z_p"])
        K1s, K2s, K2p = qf.k_two_3s(element["coeff_1s"], element["coeff_2s"], element["coeff_2p"], element["z_s"], element["z_p"])
    else:
        R1s, R2s, R2p = qf.R_one_3s(element["coeff_1s"], element["coeff_2s"], element["coeff_2p"], element["z_s"], element["z_p"])
        K1s, K2s, K2p = qf.k_one_3s(element["coeff_1s"], element["coeff_2s"], element["coeff_2p"], element["z_s"], element["z_p"])

    t = []

    p_R = 1 / (4 * np.pi * element["Z"]) * (element["a"] * R1s**2 + element["b"] * R2s**2 + element["c"]* R2p**2)

    qf.plot_radius(p_R, 1.3, f"DOS of {element['name']} in the position space", "r", "rho(r)", element["name"])

    S_r = qf.integrate_radius_to_inf(-4 * sp.pi * p_R * sp.log(p_R + epsilon) * r**2)
    s_R.append(S_r)


    n_k = 1 / (4 * np.pi * element["Z"]) * (element["a"] * K1s**2 + element["b"] * K2s**2 + element["c"] * K2p**2)

    t.append(['S_r', S_r])

    qf.plot_k(n_k, 3, f"DOS of {element['name']} in the k space", "k", "n(k)", element["name"])

    S_k = qf.integrate_k_to_inf(-4 * sp.pi * n_k * sp.log(n_k + epsilon) * k**2)
    s_K.append(S_k)
    t.append(['S_k', S_k])

    S_tot = S_r + S_k
    s_tot.append(S_tot)
    t.append(['S_tot', S_tot])

    df = pd.DataFrame(t, columns=["Quantity", "Result"])
    pd.options.display.float_format = '{:.12g}'.format
    print(df)
    print("\n")


qf.scatter_plot(zs, s_R, "Z", "Sr(Z)", "The entropy in the position space as a function of Z")
qf.scatter_plot(zs, s_K, "Z", "Sk(Z)", "The entropy in the k-space Sk as a function of Z")
qf.scatter_plot(zs, s_tot, "Z", "S(Z)", "The total entropy S as a function of Z")
