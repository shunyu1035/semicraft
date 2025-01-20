# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.operations.sputterYield as sputterYield
import numpy as np
import matplotlib.pyplot as plt
import os


plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


fig, ax = plt.subplots()

yield1 = sputterYield.sputter_yield_angle(0.3, 0.001, np.pi/5)
line1, = ax.plot(yield1 [1, :], yield1 [0, :], '-', label="$\gamma_0 = 0.3, \gamma_{max} = 0.001, \Theta_{max} = \pi/5$")

yield2 = sputterYield.sputter_yield_angle(0.2, 0.001, np.pi/5)
line2, = ax.plot(yield2 [1, :], yield2 [0, :], '-', label="$\gamma_0 = 0.2, \gamma_{max} = 0.001, \Theta_{max} = \pi/5$")

yield3 = sputterYield.sputter_yield_angle(0.3, 0.01, np.pi/5)
line3, = ax.plot(yield3 [1, :], yield3 [0, :], '-', label="$\gamma_0 = 0.3, \gamma_{max} = 0.01, \Theta_{max} = \pi/5$")

yield4 = sputterYield.sputter_yield_angle(0.3, 0.001, np.pi/4)
line4, = ax.plot(yield4 [1, :], yield4 [0, :], '-', label="$\gamma_0 = 0.3, \gamma_{max} = 0.005, \Theta_{max} = \pi/4$")

ax.legend(bbox_to_anchor=(1.15, 1),loc='upper right', borderaxespad=0.)
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式



