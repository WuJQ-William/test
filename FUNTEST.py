import numpy as np
from matplotlib import pyplot as plt

from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant

def Krige(X, Y, X_pre, num):
    kx = X.reshape(num, 1)
    ky = Y.reshape(num, 1)
    regression = regr_quadratic
    correlation = corr_gauss
    dacefit = DACE(regr=regression, corr=correlation,
                   theta=2, thetaL=None, thetaU=None)
    dacefit.fit(kx, ky)
    pred, _mse = dacefit.predict(X_pre[:, None], return_mse=True)
    return pred, _mse

sample = 6
x = np.linspace(0, 1, sample)
y = (6*x-2)**2*np.sin(12*x-4)

gridx = np.linspace(0,1,1001)
Funy = (6*gridx-2)**2*np.sin(12*gridx-4)
Prey, mse = Krige(x, y, gridx, sample)

max = np.empty(1001)
min = np.empty(1001)
max2 = np.empty(1001)
min2 = np.empty(1001)
tolerance = 0.1
for n in range(1001):
    Nx = np.linspace(gridx[n]-tolerance, gridx[n]+tolerance, 10000)
    Ny,mse2 = Krige(x, y, Nx, sample)
    mse = np.sqrt(mse2)
    c = [Ny+1.96*mse,Ny-1.96*mse]

    max[n] = np.max(c)
    min[n] = np.min(c)
    max2[n] = np.max(Ny)
    min2[n] = np.min(Ny)

plt.plot(gridx, Funy)
plt.scatter(x,y)
plt.plot(gridx, Prey)
#plt.plot(gridx, max)
#plt.plot(gridx, mse)
plt.fill_between(gridx,min, max, alpha =0.2)
plt.fill_between(gridx,min2, max2, alpha =0.2)
plt.show()
