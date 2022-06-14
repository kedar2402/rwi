import autograd.numpy as np   # Thinly-wrapped version of Numpy

import scipy as sp
from scipy.sparse import dia_matrix
from scipy.linalg import det

from autograd import grad

#import matplotlib.pyplot as plt

from complex_roots import RootFinderRectangle

class RWI_coeff():
    """Dispersion relation RWI: A*psi'' + B*psi' + C*psi = 0"""
    def __init__(self,
                 background_density,
                 background_pressure,
                 Gamma,
                 radius,
                 homentropic_flag=False):

        self.radius = radius              # Array of radial coordinates
        self.Sigma = background_density   # Surface density function
        self.P = background_pressure      # Pressure function

        self.G = Gamma                    # Adiabatic exponent
        self.homentropic_flag = homentropic_flag

        # Automatic differentiation of surface density and pressure
        self.dSigma = grad(self.Sigma)
        self.dP = grad(self.P)

        # Equilibrium angular velocity
        self.Omega = lambda r: np.sqrt(1/r/r/r +
                                       self.dP(r)/self.Sigma(r)/r)

        # Calculate epicyclic frequency squared
        f = lambda r: r*r*r*r*self.Omega(r)*self.Omega(r)
        dOm = grad(f)
        self.kappa2 = lambda r: dOm(r)/r/r/r

        # Radial derivative of angular velocity
        self.dOmega = grad(self.Omega)
        # Radial derivative of epicyclic frequency squared
        self.dkappa2 = grad(self.kappa2)

        # Sound speed squared
        self.cs2 = lambda r: self.G*self.P(r)/self.Sigma(r)

        # Radial derivative of cs2/Ls/Lp
        f = lambda r: self.cs2(r)*self.invLs(r)/self.Lp(r)
        self.dcs2 = grad(f)

        self.precompute()

    def precompute(self):
        """Precompute as many arrays over r as possible"""
        r = self.radius

        self._O = 0.0*r
        self._dO = 0.0*r
        self._k2 = 0.0*r
        self._dk2 = 0.0*r
        self._iLs = 0.0*r
        self._Lp = 0.0*r
        self._cs2 = 0.0*r
        self._dcs2 = 0.0*r
        self._Sig = 0.0*r
        self._dSig = 0.0*r
        self._dFL = 0.0*r

        # Automatic differentiation
        funcL = lambda x: x*self.invLs(x)/self.Omega(x)
        dFL = grad(funcL)

        for i in range(0, len(r)):
            self._O[i]    = self.Omega(r[i])
            self._dO[i]   = self.dOmega(r[i])
            self._k2[i]   = self.kappa2(r[i])
            self._dk2[i]  = self.dkappa2(r[i])
            self._iLs[i]  = self.invLs(r[i])
            self._Lp[i]   = self.Lp(r[i])
            self._cs2[i]  = self.cs2(r[i])
            self._dcs2[i] = self.dcs2(r[i])
            self._Sig[i]  = self.Sigma(r[i])
            self._dSig[i] = self.dSigma(r[i])
            self._dFL[i]  = dFL(r[i])

    def invLs(self, r):
        if self.homentropic_flag == True:
            return 0.0*r

        denom = self.dP(r)/self.P(r) - self.G*self.dSigma(r)/self.Sigma(r)
        return denom/self.G

    def Lp(self, r):
        denom = self.dP(r)/self.P(r)
        return self.G/denom

    def _F(self, dw, m):
        denom = self._k2 - dw*dw - self._cs2*self._iLs/self._Lp
        return self._Sig*self._O/denom

    def _dF(self, dw, m):
        """Radial derivative of F"""

        # Radial derivative of dw*dw
        dw2 = -2*dw*m*self._dO

        denom = self._k2 - dw*dw - self._cs2*self._iLs/self._Lp
        num = (self._dSig*self._O + self._Sig*self._dO)*denom -\
               self._Sig*self._O*(self._dk2 - dw2 - self._dcs2)

        return num/denom/denom

    def __call__(self, omega, m):
        dw = omega - m*self._O   # Shifted frequency
        F = self._F(dw, m)       # F function from Lovelace et al. (1998)
        dF = self._dF(dw, m)     # Radial derivative of F
        kp = m/self.radius       # Angular wave number

        # Coefficient of second derivative of Psi
        A = F/self._O

        # derivative of r*F/Omega
        num = (self.radius*dF + F)*self._O - self.radius*F*self._dO
        der = num/self._O/self._O
        # Coefficient of first derivative of Psi
        B = der/self.radius

        # Coefficient of zeroth derivative of Psi
        C = -kp*kp*F/self._O + 0.0*1j
        C -= self._Sig/self._cs2
        C -= 2*kp*dF/dw
        C -= F*self._iLs*self._iLs/self._O
        C -= (self._dFL*F + self.radius*dF*self._iLs/self._O)/self.radius
        C -= 4*kp*F*self._iLs/dw
        C -= kp*kp*self._cs2*F*self._iLs/dw/dw/self._O/self._Lp

        return A, B, C

class RWI():
    def __init__(self, r, RWI_coeff):
        self.r = r
        self.dr = r[1] - r[0]

        self.N = len(r)

        self.coeff = RWI_coeff

    def detM(self, omega, m):
        A, B, C = self.coeff(omega, m)

        a = C - 2*A/self.dr**2
        a[0] = C[0] - A[0]/self.dr**2 - 0.5*B[0]/self.dr
        a[-1]= C[-1] - A[-1]/self.dr**2 + 0.5*B[-1]/self.dr

        b = A/self.dr**2 + 0.5*B/self.dr
        c = np.roll(A/self.dr**2 - 0.5*B/self.dr, -1)

        #print('a = ', a)

        f = a + 0.0*1j
        f[1] = a[1]*f[0] - c[0]*b[0]

        for n in range(2, len(self.r)):
            f[n] = a[n]*f[n-1] - c[n-1]*b[n-1]*f[n-2]

        return f[-1]

    def construct_matrix(self, omega, m):
        """Construct tridiagonal sparse matrix"""

        A, B, C = self.coeff(omega, m)

        diag = C - 2*A/self.dr**2
        diag[0] = C[0] - A[0]/self.dr**2 - 0.5*B[0]/self.dr
        diag[-1]= C[-1] - A[-1]/self.dr**2 + 0.5*B[-1]/self.dr

        diag_up = np.roll(A/self.dr**2 + 0.5*B/self.dr, 1)
        diag_down = np.roll(A/self.dr**2 - 0.5*B/self.dr, -1)

        data = np.array([diag_down, diag, diag_up])
        offsets = np.array([-1, 0, 1])

        return dia_matrix((data, offsets), shape=(self.N, self.N))

        # A*(PsiR - 2*PsiM + PsiL)/dr^2 + B*(PsiR - PsiL)/(2*dr) + C*PsiM = 0

        # Left boundary: PsiL = PsiM
        # A*(PsiR - PsiM)/dr^2 + B*(PsiR - PsiM)/(2*dr) + C*PsiM = 0

        # Right boundary: PsiR = PsiM
        # A*(PsiL - PsiM)/dr^2 + B*(PsiM - PsiL)/(2*dr) + C*PsiM = 0

        # Diagonal:
        # i=1: C(r[i]) - A(r[i])/dr^2 - 0.5*B(r[i])/dr
        # i,i: C(r[i]) - 2*A(r[i])/dr^2
        # i=N: C(r[i]) - A(r[i])/dr^2 + 0.5*B(r[i])/dr

        # Off-diagonal i,i+1: A(r[i])/dr^2 + 0.5*B(r[i])/dr
        # Off-diagonal i,i-1: A(r[i])/dr^2 - 0.5*B(r[i])/dr

    def func(self, omega, m, logmaxdet=0):
        """Return scales determinant of matrix M"""
        # Make sure we can handle scalar and vector input.
        omega = np.asarray(omega)
        scalar_input = False
        if omega.ndim == 0:
            omega = omega[None]  # Makes omega 1D
            scalar_input = True
        else:
            original_shape = np.shape(omega)
            omega = np.ravel(omega)

        logdet = 0.0*omega
        sgn = 0.0*omega
        for i in range(0, len(omega)):
            #print(i)
            M = self.construct_matrix(omega[i], m)
            sgn[i], logdet[i] = np.linalg.slogdet(M.toarray())

        if logmaxdet == 0:
            logmaxdet = np.max(logdet)
            print(logmaxdet)

        ret = sgn*np.exp(logdet - logmaxdet)

        if scalar_input:
            return np.squeeze(ret)

        return np.reshape(ret, original_shape)

    def determine_scaling(self, omega, m):
        """Find the maximum of logdet of matrix M"""
        # Make sure we can handle scalar and vector input.
        omega = np.asarray(omega)
        scalar_input = False
        if omega.ndim == 0:
            omega = omega[None]  # Makes omega 1D
            scalar_input = True
        else:
            original_shape = np.shape(omega)
            omega = np.ravel(omega)

        logdet = 0.0*omega
        sgn = 0.0*omega
        for i in range(0, len(omega)):
            #print(i)
            M = self.construct_matrix(omega[i], m)
            sgn[i], logdet[i] = np.linalg.slogdet(M.toarray())

        return np.max(logdet)

    def find_root(self, m):
        real_range = [0.95*m, 1.05*m]
        imag_range = [0.001, 0.25]

        # Determine scaling needed for determinant
        x = np.linspace(real_range[0], real_range[1], 10)
        y = np.linspace(imag_range[0], imag_range[1], 10)
        xv, yv = np.meshgrid(x, y)
        omega = xv + 1j*yv
        logmaxdet = self.determine_scaling(omega, m)

        f = lambda z: self.func(z, m, logmaxdet=logmaxdet)

        root_finder = RootFinderRectangle(real_range,
                                          imag_range,
                                          n_sample=30,
                                          max_zoom_domains=10,
                                          verbose_flag=False,
                                          tol=1.0e-13,
                                          clean_tol=1.0e-4,
                                          max_secant_iterations=100)

        ret = root_finder.calculate(f)

        #for omega in ret:
        #    M = self.construct_matrix(omega, m).toarray()
        #    abseig = np.abs(sp.linalg.eigvals(M))
        #    print(np.min(abseig)/np.max(abseig))

        return ret
