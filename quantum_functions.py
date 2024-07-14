"""
Bompotas Christos
AEM:4435
This code snippet contains all the necessary functions 
that takes the coefficients from the given paper and returns the entropies
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
import pandas as pd

# Define symbolic variables
r = sp.symbols("r")
k = sp.symbols("k")

# Define radial functions
def Sr1s(z, r):
	return 2 * z**(3/2) * sp.exp(-z * r)

def Sr2s(z, r):
	return 2 / sp.sqrt(3) * z**(5/2) * r * sp.exp(-z * r)

def Sr3s(z, r):
	return 2**(3/2) / (3 * sp.sqrt(5)) * z**(7/2) * r**2 * sp.exp(-z * r)

def Sr2p(z, r):
	return 2 / sp.sqrt(3) * z**(5/2) * r * sp.exp(-z * r)

def calculate_R(element, coeff_1s, coeff_2s, coeff_2p, z_s, z_p):
	if element in ["He"]:
		ranges_1s = [slice(0, 1), slice(1, 2), slice(2, None)]
		ranges_2s = [slice(0, 1), slice(1, 2), slice(2, None)]
	elif element in ["Be", "B"]:
		ranges_1s = [slice(0, 2), slice(2, 4), slice(4, None)]
		ranges_2s = [slice(0, 2), slice(2, 4), slice(4, None)]
	else:
		ranges_1s = [slice(0, 2), slice(2, 3), slice(3, None)]
		ranges_2s = [slice(0, 2), slice(2, 3), slice(3, None)]

	R1s = sum(c * Sr1s(z, r) for c, z in zip(coeff_1s[ranges_1s[0]], z_s[ranges_1s[0]])) + \
		  sum(c * Sr3s(z, r) for c, z in zip(coeff_1s[ranges_1s[1]], z_s[ranges_1s[1]])) + \
		  sum(c * Sr2s(z, r) for c, z in zip(coeff_1s[ranges_1s[2]], z_s[ranges_1s[2]]))
	
	R2s = sum(c * Sr1s(z, r) for c, z in zip(coeff_2s[ranges_2s[0]], z_s[ranges_2s[0]])) + \
		  sum(c * Sr3s(z, r) for c, z in zip(coeff_2s[ranges_2s[1]], z_s[ranges_2s[1]])) + \
		  sum(c * Sr2s(z, r) for c, z in zip(coeff_2s[ranges_2s[2]], z_s[ranges_2s[2]]))
	
	R2p = sum(c * Sr2p(z, r) for c, z in zip(coeff_2p, z_p))
	
	return R1s, R2s, R2p

# Define momentum space functions
def k_1s(z, k):
	return 1 / (2 * sp.pi)**(3/2) * 16 * sp.pi * z**(5/2) / (z**2 + k**2)**2

def k_2s(z, k):
	return 1 / (2 * sp.pi)**(3/2) * 16 * sp.pi * z**(5/2) * (3 * z**2 - k**2) / (sp.sqrt(3) * (z**2 + k**2)**3)

def k_3s(z, k):
	return 1 / (2 * sp.pi)**(3/2) * 64 * sp.sqrt(10) * sp.pi * z**(9/2) * (z**2 - k**2) / (5 * (z**2 + k**2)**4)

def k_2p(z, k):
	return 1 / (2 * sp.pi)**(3/2) * 64 * sp.pi * k * z**(7/2) / (sp.sqrt(3) * (z**2 + k**2)**3)

def calculate_K(element, coeff_1s, coeff_2s, coeff_2p, z_s, z_p):
	if element in ["He"]:
		ranges_1s = [slice(0, 1), slice(1, 2), slice(2, None)]
		ranges_2s = [slice(0, 1), slice(1, 2), slice(2, None)]
	elif element in ["Be", "B"]:
		ranges_1s = [slice(0, 2), slice(2, 4), slice(4, None)]
		ranges_2s = [slice(0, 2), slice(2, 4), slice(4, None)]
	else:
		ranges_1s = [slice(0, 2), slice(2, 3), slice(3, None)]
		ranges_2s = [slice(0, 2), slice(2, 3), slice(3, None)]

	K1s = sum(c * k_1s(z, k) for c, z in zip(coeff_1s[ranges_1s[0]], z_s[ranges_1s[0]])) + \
		  sum(c * k_3s(z, k) for c, z in zip(coeff_1s[ranges_1s[1]], z_s[ranges_1s[1]])) + \
		  sum(c * k_2s(z, k) for c, z in zip(coeff_1s[ranges_1s[2]], z_s[ranges_1s[2]]))
	
	K2s = sum(c * k_1s(z, k) for c, z in zip(coeff_2s[ranges_2s[0]], z_s[ranges_2s[0]])) + \
		  sum(c * k_3s(z, k) for c, z in zip(coeff_2s[ranges_2s[1]], z_s[ranges_2s[1]])) + \
		  sum(c * k_2s(z, k) for c, z in zip(coeff_2s[ranges_2s[2]], z_s[ranges_2s[2]]))
	
	K2p = sum(c * k_2p(z, k) for c, z in zip(coeff_2p, z_p))
	
	return K1s, K2s, K2p

# Function to integrate symbolic expressions of r from 0 to infinity
def integrate_radius_to_inf(f):
	num_f = sp.lambdify(r, f, modules=["numpy"])
	def integrand(r):
		return num_f(r)
	int_result, error_result = quad(integrand, 0, np.inf)
	return int_result

# Function to plot a symbolic function from r 0 to rmax
def plot_radius(f, title, xlabel, ylabel, element_name):
	sns.set(style="whitegrid")  

	r = sp.symbols('r')
	num_f = sp.lambdify(r, f, modules=["numpy"])
	
	radius_space = np.linspace(0, 6, 10000)
	f_space = num_f(radius_space)
	
	plt.figure(figsize=(10, 6))
	plt.plot(radius_space, f_space, label=f'{element_name}', color='b', linewidth=2)
	
	plt.title(title, fontsize=20, fontweight='bold')
	plt.xlabel(xlabel, fontsize=15)
	plt.ylabel(ylabel, fontsize=15)
	plt.legend(loc='best', fontsize=12)
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.minorticks_on()
	plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
	plt.xlim(0, 6)
	#plt.yticks(np.arange(0, 60, 5))
	plt.savefig(f"rden{element_name}.png")
	plt.show()

# Function to integrate symbolic expressions of k from 0 to infinity
def integrate_k_to_inf(f):
	num_f = sp.lambdify(k, f, modules=["numpy"])
	def integrand(k):
		return num_f(k)
	int_result, error_result = quad(integrand, 0, np.inf)
	return int_result

# Function to plot a symbolic function from k 0 to kmax
def plot_k(f, title, xlabel, ylabel, element_name):
	sns.set(style="whitegrid")

	k = sp.symbols('k')
	num_f = sp.lambdify(k, f, modules=["numpy"])
	
	k_space = np.linspace(0, 7, 10000)
	f_space = num_f(k_space)
	
	plt.figure(figsize=(10, 6))
	plt.plot(k_space, f_space, label=f'{element_name}', color='b', linewidth=2)
	
	plt.title(title, fontsize=20, fontweight='bold')
	plt.xlabel(xlabel, fontsize=15)
	plt.ylabel(ylabel, fontsize=15)
	plt.legend(loc='best', fontsize=12)
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.minorticks_on()
	plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
	plt.xlim(0,7)
	plt.savefig(f"kden{element_name}.png")
	plt.show()

def analyze_elements(elements):
	names=[]
	epsilon = 10**(-17)
	s_R = []
	s_K = []
	s_tot = []
	zs = []

	r = sp.symbols("r")
	k = sp.symbols("k")
	
	def process_element(element):
		R1s, R2s, R2p = calculate_R(element["name"], element["coeff_1s"], element["coeff_2s"], element["coeff_2p"], element["z_s"], element["z_p"])
		K1s, K2s, K2p = calculate_K(element["name"], element["coeff_1s"], element["coeff_2s"], element["coeff_2p"], element["z_s"], element["z_p"])

		p_R = 1 / (4 * np.pi * element["Z"]) * (element["a"] * R1s**2 + element["b"] * R2s**2 + element["c"]* R2p**2)
		plot_radius(p_R*4*np.pi*r**2, f"Electron Probability Density of {element['name']} in the position space", "r", "ρ(r)", element["name"])

		S_r = integrate_radius_to_inf(-4 * sp.pi * p_R * sp.log(p_R + epsilon) * r**2)
		s_R.append(S_r)

		n_k = 1 / (4 * np.pi * element["Z"]) * (element["a"] * K1s**2 + element["b"] * K2s**2 + element["c"] * K2p**2)
		plot_k(n_k*4*np.pi*k**2, f"Electron Probability Density of {element['name']} in the k space", "k", "n(k)", element["name"])

		S_k = integrate_k_to_inf(-4 * sp.pi * n_k * sp.log(n_k + epsilon) * k**2)
		s_K.append(S_k)

		S_tot = S_r + S_k
		s_tot.append(S_tot)

		zs.append(element["Z"])
		names.append(element["name"])
		df = pd.DataFrame([["S_r", S_r], ["S_k", S_k], ["S_tot", S_tot]], columns=["Entropies", "Calculation"])
		pd.options.display.float_format = '{:.12g}'.format
		print(f"About {element['name']}:")
		print(df)
		print("\n")

	for element in elements:
		process_element(element)

	scatter_plot(zs, s_R, "Z", "Sr(Z)", "The entropy in the position space as a function of Z", names)
	scatter_plot(zs, s_K, "Z", "Sk(Z)", "The entropy in the k-space as a function of Z", names)
	scatter_plot(zs, s_tot, "Z", "S(Z)", "The total entropy as a function of Z", names)

def scatter_plot(x, y, xlabel, ylabel, title, labels):
	plt.figure(figsize=(10, 6))
	plt.scatter(x, y, color='b')
	for i, label in enumerate(labels):
		plt.text(x[i], y[i], label, fontsize=15, ha='center',va='bottom')
	plt.title(title, fontsize=20, fontweight='bold')
	plt.xlabel(xlabel, fontsize=15)
	plt.ylabel(ylabel, fontsize=15)
	plt.grid(True, linestyle='--', linewidth=0.5)
	plt.savefig(f"{ylabel}.png")
	plt.show()
