import calibration_simplefilm_GP_1003_fitting_o2
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective
from skopt.callbacks import CheckpointSaver
from skopt import load

import os
import matplotlib.pyplot as plt


path = "calibration_1014_paper_gpmin"
# path = "calibration_1008_paper_fit"

folder_path = f"./{path}/hyperopt"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"文件夹已创建：{folder_path}")
else:
    print(f"文件夹已存在：{folder_path}")


# pertubation = 0.5

# # react_prob_chemical_0 = 0.73
# # react_prob_chemical_1 = 0.61
# # react_prob_chemical_10 = 0.03911
# # react_prob_chemical_11 = 0.11244
# # reflect_coefficient = 0.16
# # react_yield_p0_2 = 0.6
# # react_yield_p0_3 = 0.2

# react_prob_chemical_0 = 0.5
# react_prob_chemical_1 = 0.5
# react_prob_chemical_10 = 0.1
# react_prob_chemical_11 = 0.1
# reflect_coefficient = 0.3
# react_yield_p0_2 = 0.5
# react_yield_p0_3 = 0.3


# def min(param, pertubation):
#     return param - param*pertubation

# def max(param, pertubation):
#     return param + param*pertubation




# 定义参数空间（顺序要和 objective 里的参数顺序一致）
# space = [
#     Real(min(react_prob_chemical_0, pertubation), max(react_prob_chemical_0, pertubation), name='pF_Si'),
#     Real(min(react_prob_chemical_1, pertubation), max(react_prob_chemical_1, pertubation), name='pF_SiF'),
#     Real(min(react_prob_chemical_10, pertubation*1.5), max(react_prob_chemical_10, pertubation*1.5), name='pO_Si'),
#     Real(min(react_prob_chemical_11, pertubation*1.5), max(react_prob_chemical_11, pertubation*1.5), name='pO_SiF'),
#     Real(min(reflect_coefficient, pertubation), max(reflect_coefficient, pertubation), name='Rco'),
#     Real(min(react_yield_p0_2, pertubation), max(react_yield_p0_2, pertubation), name='pIon_SiOF'),
#     Real(min(react_yield_p0_3, pertubation), max(react_yield_p0_3, pertubation), name='pIon_mask')
# ]

space = [
    Real(0.3, 1, name='pF_Si'),
    Real(0.3, 1, name='pF_SiF'),
    Real(0.005, 0.2, name='pO_Si'),
    Real(0.005, 0.2, name='pO_SiF'),
    Real(0.1, 0.9, name='Rco_atom'),
    # Real(0.1, max(react_yield_p0_2, pertubation), name='pIon_SiOF'),
    # Real(0.01, 0.3, name='pIon_mask'),
    Real(0.1, 0.8, name='gamma_0_SiOF'),
    Real(1, 5, name='f_SiOF'),
    Real(0.1, 1.4, name='theta_max_SiOF')
]


param_names = [
    'react_prob_chemical_0',
    'react_prob_chemical_1',
    'react_prob_chemical_10',
    'react_prob_chemical_11',
    'reflect_coefficient',
    # 'react_yield_p0_2',
    # 'react_yield_p0_3',
    'gamma_0',
    'f',
    'theta_max'
]


# 自定义目标函数（包含 iter 计数并保存参数到 .npy）
def objective(params):
    # 将参数列表转为 dict，便于后续代码兼容
    params_dict = dict(zip(param_names, params))
    iter_num = len(os.listdir(folder_path))  # 用已保存文件数计数
    loss = calibration_simplefilm_GP_1003_fitting_o2.trainProfile(iter_num, params_dict, path)

    # 保存参数 & loss 到 .npy 文件
    save_data = {'iter': iter_num, 'params': params_dict, 'loss': loss}
    np.save(f'./{path}/hyperopt/iteration_{iter_num}.npy', save_data)  # 以 iter_num 命名文件

    print(f"Iteration {iter_num}: Saved iteration_{iter_num}.npy | Loss: {loss}")
    
    return 1 - loss  # skopt 也需要最小化目标



step = 300

checkpoint_saver = CheckpointSaver(f"./{path}/checkpoint.pkl", compress=9) # keyword arguments will be passed to `skopt.dum

# 运行 skopt 的高斯过程优化
res = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=step,
    n_random_starts=3,  # the number of random initial points
    callback=[checkpoint_saver],
    random_state=777
)



ax = plot_evaluations(res, bins=10)
plt.savefig(f"./{path}/skopt_evaluations_{step}.png", dpi=300, bbox_inches='tight')
plt.close()

ax = plot_objective(res)
plt.savefig(f"./{path}/objective_plot_{step}.png", dpi=300, bbox_inches='tight')
plt.close()





# step = 10

# checkpoint_saver = CheckpointSaver(f"./{path}/checkpoint.pkl", compress=8) # keyword arguments will be passed to `skopt.dum

# # 运行 skopt 的高斯过程优化
# res = gp_minimize(
#     func=objective,
#     dimensions=space,
#     n_calls=step,
#     n_random_starts=3,  # the number of random initial points
#     callback=[checkpoint_saver],
#     random_state=777
# )



# ax = plot_evaluations(res, bins=10)
# plt.savefig(f"./{path}/skopt_evaluations_{step}.png", dpi=300, bbox_inches='tight')
# plt.close()

# ax = plot_objective(res)
# plt.savefig(f"./{path}/objective_plot_{step}.png", dpi=300, bbox_inches='tight')
# plt.close()


# res = load(f"./{path}/checkpoint.pkl")
# x0 = res.x_iters
# y0 = res.func_vals

# total_steps = 300
# interval = 10

# for i in range(step, step + total_steps, interval):
#     n_calls = interval
#     res = gp_minimize(
#         func=objective,
#         dimensions=space,
#         n_calls=n_calls,
#         x0=x0,
#         y0=y0,
#         n_random_starts=3,  # the number of random initialization points
#         callback=[checkpoint_saver],
#         random_state=777
#     )
#     # 只追加新点
#     res = load(f"./{path}/checkpoint.pkl")
#     x0 = res.x_iters
#     y0 = res.func_vals

#     # 绘图
#     ax = plot_evaluations(res, bins=10)
#     plt.savefig(f"./{path}/skopt_evaluations_{i + n_calls}.png", dpi=300, bbox_inches='tight')
#     plt.close()


#     ax = plot_objective(res)
#     plt.savefig(f"./{path}/objective_plot_{i + n_calls}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"已完成{i + n_calls}个samples, 结果已保存。")




# steps = 100

# # 运行 skopt 的高斯过程优化
# res = gp_minimize(
#     func=objective,
#     dimensions=space,
#     n_calls=steps,
#     random_state=0
# )

# step = 130
# x0 = np.load(f'./{path}/x_iters_{step}.npy', allow_pickle=True)
# y0 = np.load(f'./{path}/func_vals_{step}.npy', allow_pickle=True)
# x0 = x0.tolist()
# y0 = y0.tolist()
# # print("x0:", x0.shape)
# # print("y0:", y0.shape)

# res = gp_minimize(
#     func=objective,
#     dimensions=space,
#     n_calls=steps,
#     x0=x0,
#     y0=y0,
#     random_state=0
# )


# step2 = step + steps

# np.save(f'./{path}/x_iters_{step2}.npy', res.x_iters)
# np.save(f'./{path}/func_vals_{step2}.npy', res.func_vals)


# ax = plot_evaluations(res, bins=10)
# plt.savefig(f"./{path}/skopt_evaluations_{step2}.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("\n最优参数:", dict(zip(param_names, res.x)))