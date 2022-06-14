import autograd.numpy as np   # Thinly-wrapped version of Numpy

import rwi

np.random.seed(0)

# Homogeneous Gaussian Bump (Li et al. 2000)
A = 1.2
dr = 0.05
h = 0.1
Gamma = 5/3

Sigma = lambda r: 1.0 + (A-1)*np.exp(-0.5*(r - 1)**2/dr/dr)
Pressure = lambda r: h*h*np.power(Sigma(r), Gamma)/Gamma

r = np.linspace(0.4, 1.6, 128)
rwi_coeff = rwi.RWI_coeff(Sigma, Pressure, 5/3, r, homentropic_flag=True)
rwi = rwi.RWI(r, rwi_coeff)

# Compare right panels of fig 9 from Li et al. (2000)
m = 5
root = rwi.find_root(m)[0]

print('Mode frequency: {}, growth rate: {}'.format(np.real(root)/m,
                                                   np.imag(root)))
