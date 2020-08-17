import numpy as np
from matplotlib import pyplot as plt

from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant
sample = 10
x = np.linspace(0, 1, sample)
y = (6*x-2)**2*np.sin(12*x-4)
gridx = np.linspace(0,1,101)
Func = (6*gridx-2)**2*np.sin(12*gridx-4)

kx = x.reshape(sample, 1)
ky = y.reshape(sample, 1)
regression = regr_quadratic
correlation = corr_gauss
dacefit = DACE(regr=regression, corr=correlation,
               theta=2, thetaL=None, thetaU=None)
dacefit.fit(kx, ky)
pred, _mse= dacefit.predict(gridx[:,None], return_mse=True)

plt.plot(gridx, Func)
plt.plot(gridx, pred)
plt.plot(gridx, _mse)
plt.show()
