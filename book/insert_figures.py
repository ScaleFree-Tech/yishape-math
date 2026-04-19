#!/usr/bin/env python3
"""批量更新教材markdown文件的图引用和图表说明"""
import re, os

base = "/home/reremouse/work/yishape-math/book/"

figures = [
    ("fig_1_1_vector_ops.png", "Chapter1_LinearAlgebra/1.1.Vector.md", "向量运算", "向量加法、数乘、内积的几何意义"),
    ("fig_1_2_matrix_transform.png", "Chapter1_LinearAlgebra/1.2.Matrix.md", "矩阵变换", "旋转、剪切、缩放、镜像、投影等基本矩阵变换"),
    ("fig_1_3_determinant_geom.png", "Chapter1_LinearAlgebra/1.2.Matrix.md", "行列式几何", "行列式的几何意义：平行四边形的面积"),
    ("fig_1_4_eigen_geom.png", "Chapter1_LinearAlgebra/1.3.Matrix decomposition.md", "特征值几何", "特征向量方向的拉伸变换，几何理解特征分解"),
    ("fig_1_5_matrix_rank.png", "Chapter1_LinearAlgebra/1.3.Matrix decomposition.md", "矩阵的秩", "秩=1、2、3矩阵的列/行相关性可视化"),
    ("fig_1_6_orth_proj.png", "Chapter1_LinearAlgebra/1.3.Matrix decomposition.md", "正交投影", "向量到子空间的正交投影，投影与残差正交"),
    ("fig_2_1_1_dataframe_structure.png", "Chapter2_DataFrame/2.1.md", "DataFrame结构", "DataFrame的行列结构：行为样本，列为特征"),
    ("fig_2_2_2_missing_outlier.png", "Chapter2_DataFrame/2.2.md", "缺失值与异常值", "缺失值热力图、插值修复、重复值检测"),
    ("fig_3_1_1_chart_types.png", "Chapter3_Visualization/3.1.md", "常见图表类型", "散点图、折线图、柱状图、饼图、热力图、箱线图适用场景对比"),
    ("fig_4_1_1_six_distributions.png", "Chapter4_Statistics/4.2.md", "六大分布对比", "正态分布、t分布、卡方分布、F分布、指数分布的概率密度对比"),
    ("fig_4_1_2_clt_detail.png", "Chapter4_Statistics/4.1.md", "CLT详细演示", "中心极限定理：样本量n=1/2/5/30时样本均值分布趋近正态"),
    ("fig_4_1_3_qq_plot.png", "Chapter4_Statistics/4.2.md", "Q-Q图", "Q-Q图判断数据是否服从正态分布"),
    ("fig_4_1_4_confidence_interval.png", "Chapter4_Statistics/4.1.md", "置信区间覆盖", "95%置信区间的覆盖概率演示"),
    ("fig_4_1_5_pvalue_region.png", "Chapter4_Statistics/4.3.md", "p值与拒绝域", "假设检验的拒绝域与p值，不同显著性水平下的临界值"),
    ("fig_4_1_6_type12_error.png", "Chapter4_Statistics/4.3.md", "I类与II类错误", "I类错误（α）和II类错误（β）的几何解释与检验功效"),
    ("fig_4_3_1_ab_test.png", "Chapter4_Statistics/4.3.md", "AB测试", "AB测试的原理：双样本t检验"),
    ("fig_4_3_2_multiple_testing.png", "Chapter4_Statistics/4.3.md", "多重比较校正", "Bonferroni（控制FWER）vs Benjamini-Hochberg（控制FDR）"),
    ("fig_4_4_1_anova.png", "Chapter4_Statistics/4.4.md", "方差分析", "ANOVA F检验：三组以上均值的比较"),
    ("fig_5_1_1_regression_residuals.png", "Chapter5_Machine_learning/5.1. Regression.md", "回归残差诊断", "残差vs拟合值、残差Q-Q图、残差直方图、Scale-Location图"),
    ("fig_5_1_2_poly_regression.png", "Chapter5_Machine_learning/5.1. Regression.md", "多项式回归", "欠拟合（degree=1）→适度拟合（degree=3）→过拟合（degree=15）"),
    ("fig_5_2_1_logistic_boundary.png", "Chapter5_Machine_learning/5.2. Classification.md", "逻辑回归决策边界", "线性边界 vs 多项式逻辑回归的非线性决策边界"),
    ("fig_5_2_2_knn_boundary.png", "Chapter5_Machine_learning/5.2. Classification.md", "KNN边界", "K近邻决策边界随K值变化"),
    ("fig_5_3_1_kmeans_iter_detail.png", "Chapter5_Machine_learning/5.3. Clustering.md", "K-Means迭代", "K-Means算法迭代过程：初始化→分配→更新→收敛"),
    ("fig_5_3_2_hierarchical_dendrogram.png", "Chapter5_Machine_learning/5.3. Clustering.md", "层次聚类树状图", "层次聚类的树状图（Dendrogram）"),
    ("fig_5_3_3_silhouette_k.png", "Chapter5_Machine_learning/5.3. Clustering.md", "轮廓系数选K", "轮廓系数法选择最优聚类数K"),
    ("fig_5_4_1_pca_full_process.png", "Chapter5_Machine_learning/5.4. Dimension reduction.md", "PCA降维全过程", "PCA降维：原始散点→主成分旋转→投影到PC1（2维→1维）"),
    ("fig_5_4_2_pca_svd_compare.png", "Chapter5_Machine_learning/5.4. Dimension reduction.md", "降维方法对比", "PCA（线性）vs t-SNE（非线性）降维效果对比"),
    ("fig_5_4_1_backprop_detail.png", "Chapter5_Machine_learning/5.4. Dimension reduction.md", "反向传播", "神经网络反向传播：前向传播计算损失，反向传播计算梯度"),
    ("fig_5_4_2_dropout.png", "Chapter5_Machine_learning/5.4. Dimension reduction.md", "Dropout正则化", "Dropout：训练时随机丢弃神经元，测试时使用全部神经元"),
    ("fig_5_5_1_bagging_boosting.png", "Chapter5_Machine_learning/5.5. Tree ensembles and boosting.md", "Bagging vs Boosting", "Bagging（并行，降低方差）vs Boosting（串行，降低偏差）对比"),
    ("fig_5_5_2_xgboost_tree.png", "Chapter5_Machine_learning/5.5. Tree ensembles and boosting.md", "XGBoost树结构", "XGBoost单棵树结构与特征重要性排名"),
    ("fig_6_1_1_convex_vs_nonconvex.png", "Chapter6_Optimization/6.1. Optimization.md", "凸函数vs非凸函数", "凸函数：全局最优在内部或边界 | 非凸函数：多个局部极小"),
    ("fig_6_1_2_gd_vs_newton.png", "Chapter6_Optimization/6.1. Optimization.md", "梯度下降vs牛顿法", "梯度下降（线性收敛）vs 牛顿法（二次收敛）对比"),
    ("fig_6_2_1_simplex_path.png", "Chapter6_Optimization/6.2. Linear programming.md", "单纯形法路径", "单纯形法从可行域顶点沿边移动到最优解"),
    ("fig_6_2_2_lp_dual_shadow.png", "Chapter6_Optimization/6.2. Linear programming.md", "LP对偶理论", "线性规划的几何解释：目标函数等值线与可行域顶点"),
    ("fig_6_4_1_simulated_annealing.png", "Chapter6_Optimization/6.4. Nonconvex optimization.md", "模拟退火", "模拟退火算法：高温接受差解，低温精细搜索"),
    ("fig_7_1_1_fourier_decompose.png", "Chapter7_Time_series_and_signal_processing/7.1. Time series.md", "傅里叶分解", "时域信号→频谱分析→低通滤波重构"),
    ("fig_7_1_2_smoothing_methods.png", "Chapter7_Time_series_and_signal_processing/7.1. Time series.md", "滑动平均与指数平滑", "滑动平均（MA）与指数平滑（EWMA）的平滑效果对比"),
    ("fig_7_1_3_differencing.png", "Chapter7_Time_series_and_signal_processing/7.1. Time series.md", "差分与平稳化", "一阶差分去除趋势，二阶差分去除季节性"),
    ("fig_7_2_1_filter_types.png", "Chapter7_Time_series_and_signal_processing/7.2. signal processing.md", "滤波器类型", "低通、带通、高通滤波器的频率响应与时域效果对比"),
    ("fig_7_3_1_audio_features.png", "Chapter7_Time_series_and_signal_processing/7.3. audio processing.md", "音频特征提取", "音频波形（时域）→频谱（频域）→MFCC（梅尔频率倒谱系数）"),
    ("fig_7_4_1_music_mining.png", "Chapter7_Time_series_and_signal_processing/7.4. Music mining.md", "音乐挖掘", "节拍检测、色度图（和弦识别）、音乐推荐特征矩阵"),
]

count = 0
for fname, filepath, fig_title, fig_desc in figures:
    fullpath = base + filepath
    if not os.path.exists(fullpath):
        print(f"SKIP (not found): {filepath}")
        continue
    with open(fullpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成章节编号
    chapter_match = re.search(r'Chapter(\d+)', filepath)
    section_match = re.search(r'/(\d+\.\d+)\.', filepath)
    if chapter_match and section_match:
        ch_num = chapter_match.group(1)
        sec_num = section_match.group(1)
        fig_num = f"图{ch_num}.{sec_num}"
    else:
        fig_num = f"图 {fname}"
    
    new_insert = f"\n![{fig_title}](../figures/{fname})\n*{fig_num} {fig_title}。{fig_desc}*\n"
    
    # 找到第一个 ## 标题段落后插入
    pattern = r'(## \d+\.\d+[^\n]*\n\n)'
    match = re.search(pattern, content)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + new_insert + content[insert_pos:]
        print(f"INSERT {fig_num} in {filepath}")
    else:
        content = content + "\n" + new_insert
        print(f"APPEND {fig_num} in {filepath}")
    
    with open(fullpath, 'w', encoding='utf-8') as f:
        f.write(content)
    count += 1

print(f"\nDone! {count} figures inserted.")
