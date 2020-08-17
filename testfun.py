import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant

def testfunction(_x1, _x2):
    _f = (1 - _x1 / 2 + _x1 ** 5 + _x2 ** 3) * np.exp(-_x1 ** 2 - _x2 ** 2)
    return _f

def kriging(X, Y, X_pre):
    regression = regr_quadratic
    correlation = corr_gauss
    dacefit = DACE(regr=regression, corr=correlation,
                   theta=2, thetaL=None, thetaU=None)
    dacefit.fit(X, Y)
    pred, _mse = dacefit.predict(X_pre, return_mse=True)
    return pred, _mse

# 取样
n = 5
dim = 2
x1_sample, x2_sample = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
Xsample = np.hstack((x1_sample.reshape(n**dim, 1), x2_sample.reshape(n**dim, 1)))
Ysample = testfunction(Xsample[:, 0], Xsample[:, 1])[:, None]

# 拟合测试
n_test = 50
x1_test, x2_test = np.meshgrid(np.linspace(-3, 3, n_test), np.linspace(-3, 3, n_test))
Xtest = np.hstack((x1_test.reshape(n_test**dim, 1), x2_test.reshape(n_test**dim, 1)))
ytest, mse = kriging(Xsample, Ysample, Xtest)

# 原函数绘图
n_draw = 100
x1_draw, x2_draw = np.meshgrid(np.linspace(-3, 3, n_draw), np.linspace(-3, 3, n_draw))














# n = 5
# dim = 2
# # x,y,z为二位数组
#
# x1, x2= np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
# x = np.hstack((x1.reshape(n**2,1), x2.reshape(n**2,1)))
#
# z = testfunction(x1, x2)
#
# x1_test = np.random.random_integers(-3, 3, 100)
# x2_test = np.random.random_integers(-3, 3, 100)
# z_test = testfunction(x1_test, x2_test)
#
plt.figure('3D Scatter', facecolor='lightgray')
ax3d = plt.gca(projection='3d')

#ax3d.plot_surface(x1_sample, x2_sample, testfunction(x1_sample, x2_sample), cmap='jet', alpha = 0.6)
#ax3d.plot_surface(x1_draw, x2_draw, testfunction(x1_draw, x2_draw), cmap='jet', alpha = 0.6)
# ax3d.plot_wireframe(x1_draw, x2_draw, testfunction(x1_draw, x2_draw),linewidth=0.5, cmap='jet')
# ax3d.scatter(Xsample[:, 0], Xsample[:, 1], testfunction(Xsample[:, 0], Xsample[:, 1]))
# ax3d.scatter(Xtest[:, 0], Xtest[:, 1], ytest, alpha = 0.6)
ax3d.plot_wireframe(Xtest[:, 0], Xtest[:, 1], mse, alpha = 0.6)
plt.tight_layout()
plt.show()