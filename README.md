# 咖啡搅拌的旋度、散度与混合效率：数值复现材料

Coffee-inspired mixing under an equal-dissipation constraint.

本仓库提供论文数值结果、图表和验证的可复现代码。研究采用二维规定无滑移速度场与被动标量方程；全部实验为数值实验，未经真实咖啡实验校准。

## 下载固定版本

- [v1.0.0 发布页面](https://github.com/ZD956/coffee-mixing-reproducibility/releases/tag/v1.0.0)
- [完整复现包：代码、原始浓度场、诊断与图表](https://github.com/ZD956/coffee-mixing-reproducibility/releases/download/v1.0.0/coffee-mixing-reproducibility-v1.0.0.zip)
- [下载文件的 SHA-256](https://github.com/ZD956/coffee-mixing-reproducibility/releases/download/v1.0.0/SHA256SUMS.txt)

完整包包含201组输运记录（其中13组粗网格探索）、5组制造解检验、完整场和时间序列；12组预先固定的合成初值全部报告。请下载带上述名称的附件；GitHub自动生成的Source code ZIP只包含仓库内的轻量文件，不含原始NPZ数组。解压完整包后即可复核，无需另行克隆仓库。

## 环境与快速复核

参考环境：Python3.10.11、NumPy1.26.4、SciPy1.15.3、Numba0.61.2、llvmlite0.44.0、Matplotlib3.10.9。Numba首次调用需要编译。

在解压后含work文件夹的目录执行：

```bash
python -m pip install -r work/requirements.txt
python work/test_transport.py
python work/audit_results.py
python work/check_averaging.py
python work/independent_checks.py
python work/review_analysis.py
python work/submission_checks.py
```

上述命令复核保存结果并运行短程基准，不重跑全部长期实验。`work/final_data_audit.json`、`work/mms_checks.json`和`work/submission_review_results.json`提供发布时的验证结果。

## 从头运行

```bash
python work/prepare_jobs.py
python work/extra_jobs.py
python work/prepare_review_jobs.py
python work/run_experiments.py --stage pilot --workers 4 --force
python work/run_experiments.py --stage main --workers 4 --force
python work/run_experiments.py --stage verify --workers 4 --force
python work/run_experiments.py --stage robust --workers 4 --force
python work/run_experiments.py --stage extra --workers 4 --force
python work/run_experiments.py --stage last --workers 4 --force
python work/run_experiments.py --stage supplement --workers 2 --force
python work/run_experiments.py --stage holdout --workers 4 --force
python work/run_experiments.py --stage reviewchecks --workers 4 --force
python work/run_experiments.py --stage geometry --workers 4 --force
python work/manufactured_check.py
python work/postprocess.py
python work/review_analysis.py
python work/write_results_tex.py
```

384平方网格的计算明显更慢；workers可降至1。请在单独解压的副本重跑，`--force`将替换对应数据。原存档没有后补新版本的源码散列元数据，复算请使用`--force`。作图使用Times New Roman；缺少该字体时可按本机字体修改作图配置，数值模型不依赖字体。本仓库复现计算和图表，不包含带课程署名信息的提交PDF及其排版源。

## 文件与数据字段

| 路径 | 内容 |
|---|---|
| `work/mixing_solver.py` | 流场、Gram归一化、保守空间离散和RK4 |
| `work/*_jobs.json` | 全部参数批次 |
| `work/review_initials.py` | 合成初值规则和全部12个种子 |
| `work/review_preregistered.json` | 本地预设记录；未经外部注册 |
| `work/all_runs.csv` | 全部201组输运结果索引 |
| `work/data/`（完整包） | 201对JSON和NPZ |
| `work/verification/`（完整包） | 5组制造解的精确场和计算场 |
| `work/figures/` | 论文10幅图的矢量PDF和PNG |
| `work/review_v1/mixing_solver.py` | 追加合成初值前的代码快照 |
| `PUBLIC_MANIFEST_SHA256.json` | 完整包内部文件的SHA-256 |

NPZ字段为initial、final、snapshots、x、y、mask、history、gram。新复算另有snapshot_times。history七列为时间、全域均值、全域方差、壁带相对全域均值的方差、图扩散耗散、最小浓度、最大浓度。壁带为r>0.85，wall_J以初始全域方差归一化。

原snapshots对应步号round(k*steps/4)，k=0..4，时刻为步号乘JSON中的dt。二维数组采用ij索引，画图时转置。J=var(final)/var(initial)，只在同一初值内比较协议。JSON的t95为首个保存达标时刻；论文主表由相邻记录线性插值，因此略小。

## 适用范围与版本

耗散预算是规定连续流场的二次型，不是电机实测能耗。M平均流的预算约0.6267274，B双区流预算为1；R只在整周期累计等预算。固定时间低方差与低达标耗散是不同目标。无扩散诊断可有欠分辨振荡，仅用于时间离散耗散检查。该模型不求解实体勺体动量方程、三维环流、未溶糖晶体传质或奶泡变形。

201组原始J均已从完整场复核，质量漂移<1e-10；256平方制造解相对场误差约0.09741%；两网格小周期误差拟合阶约1.0086及1.0085。不同初值出现排序逆转，未宣称全局最优。跨平台浮点末位可以不同。

提交后的元数据追加与极短时快照修复不影响201组存档参数。完整包保留原始数据及代码快照；文件清单用于验证交付完整性，不代表全部历史运行使用同一时刻的源码。复核结果与源版本说明见`work/submission_checks.json`。
