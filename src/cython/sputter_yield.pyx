# 必须在 setup.py 中声明需要的 numpy 和 math 库
cimport numpy as cnp
import numpy as np
from libc.math cimport cos, log, exp, pi, pow

# sputter_yield_angle 函数
cdef double[:,:] sputter_yield_angle(double gamma0, double gammaMax, double thetaMax):
    """
    计算溅射产率角分布的数组
    """
    cdef double f, s, theta
    cdef int n = int(pi / 2 / 0.01)
    cdef cnp.ndarray[double, ndim=1] sputterYield = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] theta_array = np.linspace(0, pi / 2, n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] yield_hist = np.zeros((2, n), dtype=np.float64)

    f = -log(gammaMax / gamma0) / (log(cos(gammaMax)) + 1 - cos(thetaMax))
    s = f * cos(thetaMax)

    for i in range(n - 1):  # 遍历角度
        theta = theta_array[i]
        sputterYield[i] = gamma0 * pow(cos(theta), -f) * exp(-s * (1 / cos(theta) - 1))

    sputterYield[-1] = 0  # 最后一位设置为 0
    yield_hist[0, :] = sputterYield
    yield_hist[1, :] = theta_array
    return yield_hist

# sputter_yield_energy 函数
cdef inline double sputter_yield_energy(double E, double Eth):
    """
    溅射产率能量依赖
    """
    return pow(E, 0.5) - pow(Eth, 0.5)


sputter_yield_coefficient = [0.3, 0.001, np.pi/4]
sputterYield_ion = sputter_yield_angle(sputter_yield_coefficient[0], sputter_yield_coefficient[1], sputter_yield_coefficient[2])

# sputter_yield 函数
cdef double sputter_yield(double p0, double theta, double energy, double Eth):
    """
    计算溅射产率，综合角度和能量依赖
    """
    cdef:
        double interpolated_value
        double energy_yield
    interpolated_value = np.interp(theta, sputterYield_ion[1], sputterYield_ion[0])
    energy_yield = sputter_yield_energy(energy, Eth)
    return p0 * interpolated_value * energy_yield
