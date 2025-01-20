# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.operations.Rn_coeffcient as Rn_coeffcient
import numpy as np
import matplotlib.pyplot as plt
import os


plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


fig, ax = plt.subplots()

rn_prob = Rn_coeffcient.Rn_probability()
rn_angle = np.arange(0, np.pi/2, 0.01)

line1, = ax.plot(rn_angle, rn_prob, '-', label="$c_1 = 0.9423, c_2 = 0.9434, c_3 = 2.342, c_4 = 3.026$")
ax.legend(loc='lower left', borderaxespad=0.)

fig.show()
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式



