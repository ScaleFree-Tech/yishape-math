#!/usr/bin/env python3
"""更新所有markdown文件，添加新图的引用和caption"""
import re, os

BASE = '/home/reremouse/work/yishape-math/book/'

# (文件名, 章节, 图序号, 标题, 说明)
NEW_FIGS = [
    # Ch1 新增
    ('fig_1_7_1_gauss_elimination.png', 'Chapter1_LinearAlgebra/1.4.Linear Equation System.md',
     '图1.7.1 高斯消元法。高斯消元三步骤：列主元→消元→回代，复杂度O(n³)'),
    ('fig_1_8_1_ldl_cholesky.png', 'Chapter1_LinearAlgebra/1.3.Matrix decomposition.md',
     '图1.8.1 LDL分解与Cholesky分解。对称正定矩阵分解为L·D·Lᵀ或L·Lᵀ'),
    ('fig_1_9_1_condition_number.png', 'Chapter1_LinearAlgebra/1.6.Numerical stability.md',
     '图1.9.1 矩阵条件数。cond(A)越大，方程组解对扰动越敏感；病态矩阵→数值不稳定'),
    # Ch2 新增
    ('fig_2_3_1_groupby_agg.png', 'Chapter2_DataFrame/2.3.md',
     '图2.3.1 分组聚合操作。GROUP BY将数据按类别分组，聚合函数计算组统计量'),
    ('fig_2_3_2_merge_join.png', 'Chapter2_DataFrame/2.3.md',
     '图2.3.2 数据表关联。INNER/LEFT/RIGHT JOIN按键合并两张表'),
    ('fig_2_4_1_pipeline.png', 'Chapter2_DataFrame/2.4.md',
     '图2.4.1 数据清洗流水线。原始数据→缺失值处理→异常值修正→去重→类型转换→干净数据'),
    # Ch3 新增
    ('fig_3_2_1_scatter_matrix.png', 'Chapter3_Visualization/3.2.md',
     '图3.2.1 散点图矩阵。展示多维数据两两变量关系，对角线为各变量分布'),
    ('fig_3_3_1_hist_kde.png', 'Chapter3_Visualization/3.3.md',
     '图3.3.1 直方图与核密度估计。直方图近似数据分布，KDE拟合连续密度曲线'),
    # Ch4 新增
    ('fig_4_2_4_beta_gamma.png', 'Chapter4_Statistics/4.2.md',
     '图4.2.4 Beta分布与Gamma分布。Beta(α,β)描述0~1之间概率，Gamma(k,θ)描述正值计数'),
    ('fig_4_2_5_binomial_poisson.png', 'Chapter4_Statistics/4.2.md',
     '图4.2.5 二项分布与泊松分布。二项B(n,p)离散计数，泊松Poi(λ)是n大p小极限情况'),
    ('fig_4_5_1_conjugate_prior.png', 'Chapter4_Statistics/4.5.md',
     '图4.5.1 共轭先验分布。先验与后验同分布：Beta-Binomial、Normal-Normal、Dirichlet-Multinomial'),
    ('fig_4_6_1_power_test.png', 'Chapter4_Statistics/4.6.md',
     '图4.6.1 统计功效分析。功效=1-β，效应量越大、样本量越大，功效越高'),
    # Ch5 新增
    ('fig_5_1_3_ridge_path.png', 'Chapter5_Machine_learning/5.1. Regression.md',
     '图5.1.3 岭回归正则化路径。λ增大→系数收缩向0，解决多重共线性问题'),
    ('fig_5_1_4_lasso_path.png', 'Chapter5_Machine_learning/5.1. Regression.md',
     '图5.1.4 Lasso稀疏性。L1正则使某些系数精确为0，自动特征选择'),
    ('fig_5_2_3_svm_margin.png', 'Chapter5_Machine_learning/5.2. Classification.md',
     '图5.2.3 支持向量机最大间隔。SVM找最大间隔超平面，RBF核处理非线性边界'),
    ('fig_5_2_4_naive_bayes.png', 'Chapter5_Machine_learning/5.2. Classification.md',
     '图5.2.4 朴素贝叶斯概率图解。特征条件独立假设下，用贝叶斯公式计算后验概率'),
    ('fig_5_2_5_decision_tree.png', 'Chapter5_Machine_learning/5.2. Classification.md',
     '图5.2.5 决策树分裂示意。ID3/C4.5用信息增益，CART用基尼系数选择最优分裂特征'),
    ('fig_5_3_4_gmm_ellipse.png', 'Chapter5_Machine_learning/5.3. Clustering.md',
     '图5.3.4 高斯混合模型协方差椭圆。GMM用多个高斯分布混合，协方差椭圆刻画簇形状'),
    ('fig_5_3_5_dbscan.png', 'Chapter5_Machine_learning/5.3. Clustering.md',
     '图5.3.5 DBSCAN密度聚类。基于密度的聚类：核心点、边界点、噪声点，无需预设簇数'),
    ('fig_5_5_3_adaboost_weight.png', 'Chapter5_Machine_learning/5.5. Tree ensembles and boosting.md',
     '图5.5.3 AdaBoost权重更新。每轮增加被错分样本的权重，串行聚焦难样本'),
    ('fig_5_5_5_stacking.png', 'Chapter5_Machine_learning/5.5. Tree ensembles and boosting.md',
     '图5.5.5 Stacking堆叠集成。两层结构：Level 0多个基学习器，Level 1元学习器融合输出'),
    # Ch6 新增
    ('fig_6_3_1_branch_bound.png', 'Chapter6_Optimization/6.3. Integer programming.md',
     '图6.3.1 分支定界法。通过LP松弛下界和分支策略，跳过不可能的子问题，加速求解整数规划'),
    ('fig_6_3_2_cover_tsp.png', 'Chapter6_Optimization/6.3. Integer programming.md',
     '图6.3.2 集合覆盖与TSP近似。集合覆盖（NP难）用贪心近似，TSP用最近邻启发式'),
    # Ch7 新增
    ('fig_7_1_4_arima_structure.png', 'Chapter7_Time_series_and_signal_processing/7.1. Time series.md',
     '图7.1.4 ARIMA模型结构。AR(p)自回归+I差分+MA(q)移动平均，描述时间序列相关性'),
    ('fig_7_2_2_spectrogram.png', 'Chapter7_Time_series_and_signal_processing/7.2. signal processing.md',
     '图7.2.2 语谱图。时频分析工具：横轴时间，纵轴频率，颜色深浅表示功率/能量'),
    ('fig_7_2_3_wavelet.png', 'Chapter7_Time_series_and_signal_processing/7.2. signal processing.md',
     '图7.2.3 小波变换。多尺度时频分析：STFT用固定窗，小波用自适应伸缩窗，对非平稳信号更灵活'),
    ('fig_7_3_2_mfcc_bank.png', 'Chapter7_Time_series_and_signal_processing/7.3. audio processing.md',
     '图7.3.2 MFCC滤波器组。模拟人耳听觉特性的Mel刻度滤波器组→对数能量→DCT倒谱，提取语音特征'),
]

# 检查文件是否存在
for fname, fpath, caption in NEW_FIGS:
    full = BASE + fpath
    fig_full = BASE + 'figures/' + fname
    if not os.path.exists(fig_full):
        print(f"MISSING: {fname}")
    elif not os.path.exists(full):
        print(f"MISSING MD: {fpath}")

print(f"\nReady to insert {len(NEW_FIGS)} new figures")

# 插入新图
for fname, fpath, caption in NEW_FIGS:
    full = BASE + fpath
    if not os.path.exists(full):
        print(f"SKIP (no md): {fpath}")
        continue
    with open(full, encoding='utf-8') as f:
        content = f.read()
    
    # 提取章节号
    m = re.search(r'/(\d+)\.(\d+)\.', fpath)
    chapter = m.group(1) if m else '?'
    section = m.group(2) if m else '?'
    
    # 检查是否已有这个图的引用
    if fname in content:
        print(f"ALREADY in {fpath}: {fname}")
        continue
    
    # 插入图片（找到第一个 ## 标题段后）
    img_insert = f"\n![{caption.split('。')[0][3:]}](../figures/{fname})\n*{caption}*\n"
    
    # 找到第一个 ## 标题（正文开始）
    h2_match = re.search(r'\n(## \d+\.\d+[^\n]*\n)', content)
    if h2_match:
        insert_pos = h2_match.end()
        content = content[:insert_pos] + img_insert + content[insert_pos:]
    else:
        content += img_insert
    
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"INSERTED: {fname} → {fpath.split('/')[-1]}")

print("\nDone!")
