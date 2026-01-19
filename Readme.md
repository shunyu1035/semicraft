SimProfile / semicraft
=====================

本仓库包含用于论文 "SimProfile: A Monte Carlo Surface Profile Simulator with Data-Driven Parameter Calibration" 的开源代码与示例 DOI: https://doi.org/10.20944/preprints202510.1834.v1

安装
----
- 通过 Test PyPI 安装（已发布）：

	pip install -i https://test.pypi.org/simple/ semicraft==0.0.0

- 从源码安装（开发者/需要本地扩展时）：

	pip install -r requirements.txt
    pip install semicraft
	pip install -e .

快速复现论文结果
------------------
仅需运行笔记本 `SimProfile_SF6O2_paramOPT.ipynb`：

- 在 Jupyter 或 JupyterLab 中打开 `SimProfile_SF6O2_paramOPT.ipynb`，按顺序运行所有单元。该笔记本包含用于生成论文主要结果的参数与流程。

示例笔记本
-----------
- `SimProfile_Example_depo.ipynb`：演示如何设置并运行沉积（deposition）模拟。
- `SimProfile_Example_etching.ipynb`：演示如何设置并运行刻蚀（etching）模拟。


引用
----
如果您在学术工作中使用此代码，请引用：

Yao,  S. SimProfile: A Monte Carlo Surface Profile Simulator with Data-Driven Parameter Calibration. Preprints 2025, 2025101834. https://doi.org/10.20944/preprints202510.1834.v1

作者
----
姚舜禹
