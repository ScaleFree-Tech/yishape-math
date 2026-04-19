#!/usr/bin/env python3
"""精准替换每个md文件中的图片caption：替换旧的错误编号caption，
保留图片行，只更新caption行"""
import re, os

BASE = '/home/reremouse/work/yishape-math/book/'

TITLE_MAP = {
    # === Ch1 ===
    'fig_1_1_vector_ops': '向量运算',
    'fig_1_1_2d_vector': '二维向量几何',
    'fig_1_1_3d_vector': '三维向量',
    'fig_1_2_matmul': '矩阵乘法',
    'fig_1_2_matrix_ops': '矩阵运算',
    'fig_1_2_matrix_transform': '矩阵与线性变换',
    'fig_1_3_svd': '奇异值分解（SVD）',
    'fig_1_3_eigen': '特征值分解',
    'fig_1_3_determinant_geom': '行列式的几何意义',
    'fig_1_3_pca_demonstration': 'PCA主成分演示',
    'fig_1_3_svd_geometry': 'SVD几何解释',
    'fig_1_4_linear_system': '线性方程组',
    'fig_1_4_lu': 'LU分解',
    'fig_1_4_ldl': 'LDL分解',
    'fig_1_4_eigen_geom': '特征值的几何意义',
    'fig_1_5_matrix_rank': '矩阵的秩',
    'fig_1_5 QR': 'QR分解',
    'fig_1_5 least squares': '最小二乘法',
    'fig_pca_demonstration': 'PCA主成分演示',
    'fig_svd_geometry': 'SVD几何解释',
    'fig_1_6 condition': '矩阵条件数',
    'fig_1_6_gauss_jordan': '高斯-若尔当',
    'fig_1_6_iterative': '迭代法收敛',
    'fig_1_6_orth_proj': '正交投影',
    'fig_1_7_1_gauss_elimination': '高斯消元法',
    'fig_1_8_1_ldl_cholesky': 'LDL分解与Cholesky分解',
    'fig_1_9_1_condition_number': '矩阵条件数',
    # === Ch2 ===
    'fig_2_1_dataframe': 'DataFrame结构',
    'fig_2_1_1_dataframe_structure': 'DataFrame数据结构',
    'fig_2_1_column_ops': '列操作',
    'fig_2_1_indexing': '索引与切片',
    'fig_2_2_lambda': '匿名函数',
    'fig_2_2_2_missing_outlier': '缺失值与异常值处理',
    'fig_2_3_1_groupby_agg': '分组聚合操作',
    'fig_2_3_2_merge_join': '数据表关联',
    'fig_2_4_1_pipeline': '数据清洗流水线',
    'fig_2_5_statistics': '描述统计',
    # === Ch3 ===
    'fig_3_1_chart_types': '图表类型选择',
    'fig_3_1_cn': '图表类型',
    'fig_3_1_1_chart_types': '图表类型选择指南',
    'fig_3_2_1_scatter_matrix': '散点图矩阵',
    'fig_3_2_1_cn': '散点图矩阵',
    'fig_3_2 KDE': 'KDE密度估计',
    'fig_3_2_scatter': '散点图',
    'fig_3_2_cn': '散点图与分布',
    'fig_3_3_boxplot': '箱线图',
    'fig_3_3_violin': '小提琴图',
    'fig_3_3_hist_kde': '直方图与核密度估计',
    'fig_3_3_1_hist_kde': '直方图与核密度估计',
    'fig_3_3_cn': '直方图与核密度',
    'fig_3_4_corr': '相关性热力图',
    'fig_3_4_cn': '相关性热力图',
    'fig_3_5_error_bar': '误差棒',
    'fig_3_5_cn': '误差棒图',
    'fig_3_6_time_series': '时间序列图',
    'fig_3_6_cn': '时间序列图',
    # === Ch4 ===
    'fig_4_1_normal': '正态分布',
    'fig_4_1_cn': '正态分布',
    'fig_central_limit_theorem': '中心极限定理',
    'fig_4_1_6_type12_error': 'I类错误与II类错误',
    'fig_4_1_5_pvalue_region': 'P值与拒绝域',
    'fig_4_2_normal': '正态分布',
    'fig_4_2_t': 't分布',
    'fig_4_2_chi2': '卡方分布',
    'fig_4_2_f': 'F分布',
    'fig_4_2_beta_gamma': 'Beta分布与Gamma分布',
    'fig_4_2_4_beta_gamma': 'Beta分布与Gamma分布',
    'fig_4_2_5_binomial_poisson': '二项分布与泊松分布',
    'fig_4_2_3_qq_plot': 'Q-Q图正态性检验',
    'fig_4_2_1_six_distributions': '六种常用分布对比',
    'fig_distribution_comparison': '分布对比：正态、t、卡方、F',
    'fig_f_distribution': 'F分布的概率密度函数',
    'fig_4_2_clt': '中心极限定理',
    'fig_4_2_ci': '置信区间',
    'fig_4_2_cn': '常用分布',
    'fig_4_3_ab_test': 'AB测试',
    'fig_4_3_cn': 'AB测试',
    'fig_4_3_1_ab_test': 'AB测试示意图',
    'fig_4_3_2_multiple_testing': '多重比较与Bonferroni校正',
    'fig_4_1_6_type12_error': 'I类错误与II类错误',
    'fig_4_1_5_pvalue_region': 'P值与拒绝域',
    'fig_4_3_3': '统计检验流程',
    'fig_4_3_4': '功效分析',
    'fig_4_4_anova': '方差分析（ANOVA）',
    'fig_4_4_cn': '方差分析',
    'fig_4_4_1_anova': '方差分析（ANOVA）原理',
    'fig_4_5_bayes': '贝叶斯估计',
    'fig_4_5_conjugate': '共轭先验',
    'fig_4_5_1_conjugate_prior': '共轭先验分布',
    'fig_4_5_2_bayesian_update': '贝叶斯更新过程',
    'fig_bayesian_update': '贝叶斯更新过程',
    'fig_4_5_cn': '贝叶斯估计',
    'fig_4_6_power': '统计功效',
    'fig_4_6_1_power_test': '统计功效分析',
    'fig_4_6_cn': '统计功效',
    'fig_4_7 multiple_testing': '多重比较',
    'fig_4_7_cn': '多重比较',
    # === Ch5 ===
    'fig_5_1_regression': '线性回归',
    'fig_5_1_ridge': '岭回归',
    'fig_5_1_ridge_path': '岭回归正则化路径',
    'fig_5_1_lasso': 'Lasso回归',
    'fig_5_1_lasso_path': 'Lasso稀疏性',
    'fig_5_1_1_regression_residuals': '回归残差分析',
    'fig_5_1_2_poly_regression': '多项式回归与过拟合',
    'fig_5_1_3_ridge_path': '岭回归正则化路径',
    'fig_5_1_4_lasso_path': 'Lasso回归正则化路径',
    'fig_5_1_5_overfitting': '过拟合与欠拟合',
    'fig_5_1_6_bias_variance': '偏差-方差分解',
    'fig_overfitting_underfitting': '欠拟合与过拟合',
    'fig_bias_variance': '偏差-方差分解',
    'fig_5_1_cn': '线性回归',
    'fig_5_2_logistic': '逻辑回归',
    'fig_5_2_svm': '支持向量机',
    'fig_5_2_svm_margin': '支持向量机最大间隔',
    'fig_5_2_naive_bayes': '朴素贝叶斯',
    'fig_5_2_4_naive_bayes': '朴素贝叶斯概率',
    'fig_5_2_decision_tree': '决策树',
    'fig_5_2_knn': 'K近邻',
    'fig_5_2_1_logistic_boundary': '逻辑回归决策边界',
    'fig_5_2_2_knn_boundary': 'KNN决策边界可视化',
    'fig_5_2_3_svm_margin': '支持向量机最大间隔',
    'fig_5_2_4_naive_bayes': '朴素贝叶斯概率',
    'fig_5_2_5_decision_tree': '决策树分裂',
    'fig_confusion_matrix': '混淆矩阵',
    'fig_roc_curve': 'ROC曲线与AUC',
    'fig_precision_recall': '精确率-召回率曲线',
    'fig_cross_validation': 'K折交叉验证',
    'fig_5_2_cn': '分类算法',
    'fig_5_3_kmeans': 'K-means聚类',
    'fig_5_3_kmeans++': 'K-means++初始化',
    'fig_5_3_gmm': '高斯混合模型',
    'fig_5_3_gmm_ellipse': '高斯混合模型协方差椭圆',
    'fig_5_3_dbscan': 'DBSCAN',
    'fig_5_3_4_gmm_ellipse': '高斯混合模型协方差椭圆',
    'fig_5_3_5_dbscan': 'DBSCAN密度聚类',
    'fig_5_3_hierarchical': '层次聚类',
    'fig_5_3_1_kmeans_iter_detail': 'K-means迭代细节',
    'fig_5_3_2_hierarchical_dendrogram': '层次聚类树状图',
    'fig_5_3_3_silhouette_k': '轮廓系数确定最优K',
    'fig_kmeans_iteration': 'K-means聚类迭代过程',
    'fig_5_3_cn': '聚类算法',
    'fig_5_4_pca': 'PCA主成分分析',
    'fig_5_4_pca_2d': 'PCA降维2D',
    'fig_5_4_svd': 'SVD降维',
    'fig_5_4_1_pca_full_process': 'PCA完整流程',
    'fig_5_4_2_pca_svd_compare': 'PCA与SVD对比',
    'fig_5_4_2_dropout': 'Dropout正则化原理',
    'fig_5_4_1_backprop_detail': '反向传播梯度计算',
    'fig_neural_network_structure': '神经网络结构图',
    'fig_5_4_cn': '降维方法',
    'fig_5_5_bagging': 'Bagging',
    'fig_5_5_boosting': 'Boosting',
    'fig_5_5_xgboost': 'XGBoost',
    'fig_5_5_adaboost': 'AdaBoost',
    'fig_5_5_stacking': 'Stacking',
    'fig_5_5_1_bagging_boosting': 'Bagging与Boosting对比',
    'fig_5_5_2_xgboost_tree': 'XGBoost单棵决策树',
    'fig_5_5_3_adaboost_weight': 'AdaBoost权重更新',
    'fig_5_5_4': '随机森林特征重要性',
    'fig_5_5_5_ensemble_strategies': '集成学习策略汇总',
    'fig_ensemble_strategies': '集成学习策略汇总',
    'fig_5_5_5_stacking': 'Stacking堆叠集成',
    'fig_5_5_cn': '集成学习',
    'fig_5_5_ensemble': '集成学习',
    'fig_5_5_rf': '随机森林',
    'fig_5_6_deep': '深度学习',
    'fig_5_6_backprop': '反向传播',
    'fig_5_6_cn': '深度学习',
    'fig_5_ml_overview': '机器学习全景',
    # === Ch6 ===
    'fig_6_1_gd': '梯度下降',
    'fig_6_1_newton': '牛顿法',
    'fig_6_1_1_convex_vs_nonconvex': '凸优化 vs 非凸优化',
    'fig_6_1_2_gd_vs_newton': '梯度下降 vs 牛顿法',
    'fig_gradient_descent': '梯度下降法收敛过程',
    'fig_6_1_cn': '优化算法',
    'fig_6_2_lp': '线性规划',
    'fig_6_2_simplex': '单纯形法',
    'fig_6_2_dual': '对偶理论',
    'fig_6_2_1_simplex_path': '单纯形法迭代路径',
    'fig_6_2_2_lp_dual_shadow': '线性规划对偶定理与影子价格',
    'fig_6_2_cn': '线性规划',
    'fig_6_3_ip': '整数规划',
    'fig_6_3_branch_bound': '分支定界',
    'fig_6_3_1_branch_bound': '分支定界法',
    'fig_6_3_2_cover_tsp': '集合覆盖与TSP近似',
    'fig_6_3_cn': '整数规划',
    'fig_6_4_nonconvex': '非凸优化',
    'fig_6_4_sa': '模拟退火',
    'fig_6_4_1_simulated_annealing': '模拟退火算法',
    'fig_6_4_cn': '非凸优化',
    # === Ch7 ===
    'fig_7_1_ts': '时间序列',
    'fig_7_1_arima': 'ARIMA',
    'fig_7_1_arima_structure': 'ARIMA模型结构',
    'fig_7_1_1_fourier_decompose': '傅里叶分解与频谱',
    'fig_7_1_2_smoothing_methods': '时间序列平滑方法',
    'fig_7_1_3_differencing': '差分运算与平稳化',
    'fig_7_1_4_arima_structure': 'ARIMA模型结构与参数',
    'fig_7_1_acf_pacf': 'ACF与PACF',
    'fig_7_1_stationary': '平稳性检验',
    'fig_timeseries_decomposition': '时间序列分解：趋势+周期+残差',
    'fig_7_1_cn': '时间序列',
    'fig_7_2_fft': '傅里叶变换',
    'fig_7_2_filter': '滤波器',
    'fig_7_2_1_filter_types': '滤波器类型：低通/高通/带通',
    'fig_7_2_spectrogram': '语谱图',
    'fig_7_2_2_spectrogram': '语谱图',
    'fig_7_2_3_wavelet': '小波变换',
    'fig_7_2_cn': '信号处理',
    'fig_7_3_audio': '音频特征',
    'fig_7_3_mfcc': 'MFCC',
    'fig_7_3_mfcc_bank': 'MFCC滤波器组',
    'fig_7_3_1_audio_features': '音频时域与频域特征',
    'fig_7_3_2_mfcc_bank': 'MFCC滤波器组',
    'fig_7_3_cn': '音频处理',
    'fig_7_4_music': '音乐挖掘',
    'fig_7_4_1_music_mining': '音乐挖掘：节拍、旋律、和弦',
    'fig_7_4_cn': '音乐挖掘',
    # === Others ===
    'fig_acf_pacf': 'ACF与PACF',
    'fig_lp_feasible_region': '线性规划可行域',
    'fig_pca_svd': 'PCA与SVD',
    'fig_ml_concepts': '机器学习概念',
    'fig_classification_metrics': '分类指标',
    'fig_ensemble_neural': '集成与神经网络',
    'fig_timeseries_optimization': '时间序列与优化',
}

DESC_MAP = {
    'fig_1_1_vector_ops': '向量加法、数乘、点积、叉积；向量长度的几何意义',
    'fig_1_1_2d_vector': '二维平面上向量的表示及线性组合几何',
    'fig_1_1_3d_vector': '三维空间中向量的表示；右手坐标系',
    'fig_1_2_matmul': '矩阵乘法C=A·B：行·列点积；不满足交换律',
    'fig_1_2_matrix_ops': '转置、逆、行列式、迹等基本矩阵运算',
    'fig_1_2_matrix_transform': '矩阵乘法对应线性变换：旋转、缩放、剪切、投影等',
    'fig_1_3_svd': 'A=UΣVT；U、V正交；Σ对角线为奇异值；低秩近似',
    'fig_1_3_eigen': 'Ax=λx；特征值与特征向量的几何意义',
    'fig_1_3_determinant_geom': '行列式等于基向量张成平行四边形的面积（2D）或平行六面体的体积（3D）',
    'fig_1_3_pca_demonstration': 'PCA找到方差最大的方向（主成分），逐步降维',
    'fig_1_3_svd_geometry': 'SVD将矩阵分解为旋转变换+缩放变换+旋转变换的几何过程',
    'fig_1_4_linear_system': 'm个方程n个未知量；唯一解/无解/无穷多解',
    'fig_1_4_lu': 'A=LU；L单位下三角，U上三角；前代回代求解',
    'fig_1_4_ldl': 'A=LDLT；对称矩阵的LDLT分解比LU更稳定',
    'fig_1_4_eigen_geom': '特征向量是变换后方向不变的向量；特征值是缩放倍数',
    'fig_1_5_matrix_rank': '矩阵的秩=列空间维度=行空间维度；秩衡量矩阵「有效维度」',
    'fig_1_5 QR': 'A=QR；Q正交，R上三角；改善条件数',
    'fig_1_5 least squares': 'min||Ax-b||²；正规方程ATAx=ATb求最小二乘解',
    'fig_1_6_condition': 'cond(A)=||A||·||A⁻¹||；cond大→病态；扰动放大',
    'fig_1_6_gauss_jordan': '高斯-若尔当消元：化矩阵为简化行阶梯形求逆',
    'fig_1_6_iterative': '雅可比迭代与高斯-塞德尔迭代的收敛性',
    'fig_1_6_orth_proj': '向量到子空间的正交投影；残差与投影正交；最小二乘几何意义',
    'fig_1_7_1_gauss_elimination': '高斯消元通过行初等变换将增广矩阵化为上三角，再回代求解',
    'fig_1_8_1_ldl_cholesky': 'LDL分解：A=LDLᵀ；Cholesky是L=D⁰·⁵时的情况，要求A对称正定',
    'fig_1_9_1_condition_number': 'cond(A)=σmax/σmin，cond大→病态矩阵→解对舍入误差极敏感',
    'fig_2_1_dataframe': '行列结构：每列同类型，每行同实体；类Excel表格',
    'fig_2_1_1_dataframe_structure': 'DataFrame：行列索引、列类型统一、缺失值表示',
    'fig_2_1_column_ops': '列的选取、创建、删除、类型转换',
    'fig_2_1_indexing': 'iloc（位置索引）、loc（标签索引）、query（表达式）',
    'fig_2_2_lambda': '匿名函数lambda x: expr用于内联简单函数',
    'fig_2_2_2_missing_outlier': '缺失值：删除/填充/插值；异常值：IQR/Z-score检测',
    'fig_2_3_1_groupby_agg': 'GROUP BY按类别分组后用SUM/AVG/COUNT等聚合函数计算组统计量',
    'fig_2_3_2_merge_join': 'INNER JOIN只保留两边匹配的行；LEFT/RIGHT/FULL JOIN保留一侧或两侧全保留',
    'fig_2_4_1_pipeline': 'ETL流程：Extract(抽取)→Transform(清洗转换)→Load(加载)，确保数据质量',
    'fig_2_5_statistics': '均值、方差、分位数、偏度、峰度等描述统计量',
    'fig_3_1_chart_types': '根据数据类型选择合适的图表：散点图、折线图、柱状图等',
    'fig_3_1_cn': '根据数据特征选择合适的可视化图表类型',
    'fig_3_1_1_chart_types': '连续变量→直方图/箱线图；分类变量→柱状图/饼图；关系→散点图',
    'fig_3_2_1_scatter_matrix': '散点图矩阵展示所有变量对的两两关系，对角线为各变量的边际分布',
    'fig_3_2_1_cn': '散点图矩阵：多维数据两两可视化，对角线为各变量分布',
    'fig_3_2 KDE': '核密度估计用平滑核函数拟合数据的连续概率密度',
    'fig_3_2_scatter': '散点图展示两个连续变量的相关性；颜色编码第三维',
    'fig_3_2_cn': '散点图与联合分布：观察变量间的相关性',
    'fig_3_3_boxplot': '箱线图显示中位数、四分位距、须和离群点',
    'fig_3_3_violin': '小提琴图结合箱线图与核密度，同时展示分布形态和统计量',
    'fig_3_3_hist_kde': '直方图用分箱近似分布；核密度估计（KDE）用平滑核函数拟合连续密度曲线',
    'fig_3_3_1_hist_kde': '直方图用分箱近似分布；核密度估计（KDE）用平滑核函数拟合连续密度曲线',
    'fig_3_3_cn': '直方图展示数据分布；KDE拟合平滑密度曲线',
    'fig_3_4_corr': '相关性热力图用颜色深浅表示变量间相关系数矩阵',
    'fig_3_4_cn': '相关性矩阵热力图：颜色深浅表示相关性强弱',
    'fig_3_5_error_bar': '误差棒在均值点上下延伸表示置信区间或标准差',
    'fig_3_5_cn': '误差棒图：中心值为均值，棒的长度表示不确定性',
    'fig_3_6_time_series': '横轴时间、纵轴数值；折线图展示指标随时间变化',
    'fig_3_6_cn': '时间序列图：指标随时间的变化趋势和季节性模式',
    'fig_4_1_normal': '正态分布N(μ,σ²)：对称钟形曲线；μ决定位置，σ决定宽度',
    'fig_4_1_cn': '正态分布N(μ,σ²)：对称钟形；μ是均值，σ²是方差',
    'fig_central_limit_theorem': '样本均值随n增大趋近正态分布N(μ,σ²/n)，与原始分布无关',
    'fig_4_1_6_type12_error': 'I类错误（假阳性）α；II类错误（假阴性）β；功效=1-β',
    'fig_4_1_5_pvalue_region': 'P值：假设H0为真时观察到当前结果的概率；P<α则拒绝H0',
    'fig_4_2_normal': '正态分布的概率密度函数；标准正态N(0,1)转换',
    'fig_4_2_t': 't分布：尾部比正态分布更厚；n大时趋近正态',
    'fig_4_2_chi2': '卡方分布：平方和服从χ²(n)；右偏长尾分布',
    'fig_4_2_f': 'F分布：两个卡方之比；用于方差齐性检验',
    'fig_4_2_beta_gamma': 'Beta(α,β)描述0~1之间的概率；Gamma(k,θ)描述正值计数和持续时间',
    'fig_4_2_4_beta_gamma': 'Beta(α,β)描述0~1之间的概率；Gamma(k,θ)描述正值计数和持续时间',
    'fig_4_2_5_binomial_poisson': '二项B(n,p)描述n次试验成功次数；泊松Poi(λ)是n大p小极限',
    'fig_4_2_3_qq_plot': 'Q-Q图：如果点在对角线附近，则数据近似正态分布',
    'fig_4_2_1_six_distributions': '正态、t、卡方、指数、均匀、泊松等常用分布的形态对比',
    'fig_distribution_comparison': '正态分布vs t分布（尾部更厚）vs卡方（右偏）vs F分布',
    'fig_f_distribution': 'F分布是两个卡方变量的比值；用于方差分析',
    'fig_4_2_clt': '样本均值趋近正态分布；n越大近似越好',
    'fig_4_2_ci': '置信区间：重复采样时95%的区间包含真实参数',
    'fig_4_2_cn': '常用统计分布及其性质',
    'fig_4_3_ab_test': 'A/B测试：随机分组，对照组vs实验组，比较关键指标',
    'fig_4_3_cn': 'AB测试流程：分流→实验→统计检验→结论',
    'fig_4_3_1_ab_test': 'A/B测试：随机分配流量，检验实验组指标是否显著优于对照组',
    'fig_4_3_2_multiple_testing': '多次检验累积I类错误；Bonferroni：α/n； Benjamini-Hochberg控制FDR',
    'fig_4_3_3': '假设检验的标准流程：提出假设→选择统计量→计算P值→做出推断',
    'fig_4_3_4': '统计功效分析：样本量、显著性水平、效应量与功效的关系',
    'fig_4_4_anova': 'ANOVA：比较三组及以上均值是否有显著差异',
    'fig_4_4_cn': '方差分析（ANOVA）检验多组均值是否相等',
    'fig_4_4_1_anova': 'ANOVA比较K组均值；组间方差 vs 组内方差；F检验判断差异显著性',
    'fig_4_5_bayes': '贝叶斯公式：后验∝似然×先验；更新信念',
    'fig_4_5_conjugate': '共轭先验：先验与后验同分布，计算方便',
    'fig_4_5_1_conjugate_prior': '共轭先验使贝叶斯后验与先验同分布：Beta-Binomial、Normal-Normal、Dirichlet-Multinomial',
    'fig_4_5_2_bayesian_update': '贝叶斯更新：先验→新数据→后验→新的先验；信念的迭代更新',
    'fig_bayesian_update': '贝叶斯更新：先验→新数据→后验→新的先验；信念的迭代更新',
    'fig_4_5_cn': '贝叶斯估计：用数据更新先验得到后验分布',
    'fig_4_6_power': '功效=1-β：正确拒绝假阳性错误的能力',
    'fig_4_6_1_power_test': '统计功效=1-β；功效越高（通常>0.8）越容易检测到真实效应',
    'fig_4_6_cn': '统计功效分析：样本量、效应量与功效的关系',
    'fig_4_7 multiple_testing': '多重比较：Bonferroni校正控制家族错误率',
    'fig_4_7_cn': '多重比较问题：次数越多，假阳性累积越多',
    'fig_5_1_regression': '线性回归：最小二乘估计；残差分析',
    'fig_5_1_ridge': '岭回归：L2正则化；惩罚大系数；改善共线性',
    'fig_5_1_ridge_path': 'L2正则化：λ增大→系数收缩向零，解决多重共线性问题',
    'fig_5_1_lasso': 'Lasso：L1正则化；使部分系数为0（稀疏）',
    'fig_5_1_lasso_path': 'L1正则化：λ增大→部分系数精确变为零，实现稀疏特征选择',
    'fig_5_1_1_regression_residuals': '残差=实际值-预测值；残差分析检验线性假设、方差齐性、异常点',
    'fig_5_1_2_poly_regression': '多项式回归提高拟合能力，但次数过高→过拟合；需要正则化或交叉验证',
    'fig_5_1_3_ridge_path': '岭回归（L2正则）逐步收缩系数，解决多重共线性',
    'fig_5_1_4_lasso_path': 'Lasso（L1正则）使系数沿λ增大方向变为零，实现特征选择',
    'fig_5_1_5_overfitting': '欠拟合（高偏差）：模型太简单；过拟合（高方差）：模型太复杂',
    'fig_5_1_6_bias_variance': 'E[Ŷ]为预测值期望；偏差²=E[Ŷ]-Y²；方差=var(Ŷ)；噪声=var(Y)',
    'fig_overfitting_underfitting': '欠拟合（高偏差）：模型太简单；过拟合（高方差）：模型太复杂',
    'fig_bias_variance': 'E[Ŷ]为预测值期望；偏差²=E[Ŷ]-Y²；方差=var(Ŷ)；噪声=var(Y)',
    'fig_5_1_cn': '线性回归模型：拟合、自变量显著性、预测',
    'fig_5_2_logistic': '逻辑回归：sigmoid函数输出概率；二分类',
    'fig_5_2_svm': '支持向量机：最大间隔超平面；核函数处理非线性',
    'fig_5_2_svm_margin': 'SVM找最大间隔超平面使两类分开；RBF核将数据映射到高维处理非线性',
    'fig_5_2_naive_bayes': '朴素贝叶斯：特征条件独立假设；文本分类常用',
    'fig_5_2_4_naive_bayes': '特征条件独立假设下用贝叶斯公式计算后验概率，文本分类常用',
    'fig_5_2_decision_tree': 'ID3用信息增益，CART用基尼系数；每次分裂最大化类间差异',
    'fig_5_2_knn': 'K近邻：看最近的K个邻居投票；无需训练',
    'fig_5_2_1_logistic_boundary': '逻辑回归用sigmoid将线性输出转换为概率；决策边界是线性超平面',
    'fig_5_2_2_knn_boundary': 'KNN的决策边界随K增大变得平滑；K过小→过拟合，K过大→欠拟合',
    'fig_5_2_3_svm_margin': 'SVM找最大间隔超平面使两类分开；RBF核将数据映射到高维处理非线性',
    'fig_5_2_4_naive_bayes': '特征条件独立假设下用贝叶斯公式计算后验概率，文本分类常用',
    'fig_5_2_5_decision_tree': 'ID3用信息增益，CART用基尼系数；每次分裂最大化类间差异',
    'fig_confusion_matrix': 'TN/FP/FN/TP矩阵；准确率=(TP+TN)/(TP+FP+FN+TN)；精确率=TP/(TP+FP)',
    'fig_roc_curve': 'ROC曲线：TPR vs FPR在不同阈值下的表现；AUC=ROC下面积，越大越好',
    'fig_precision_recall': 'PR曲线：精确率 vs 召回率；当正负样本不平衡时比ROC更合适',
    'fig_cross_validation': 'K折交叉验证：数据分K份，轮流用K-1份训练、1份验证，减小方差',
    'fig_5_2_cn': '分类算法对比：逻辑回归、SVM、决策树、朴素贝叶斯、KNN',
    'fig_5_3_kmeans': 'K-means：随机初始化→分配簇→更新中心→迭代直至收敛',
    'fig_5_3_kmeans++': 'K-means++：用概率加权的初始化方法，改善收敛',
    'fig_5_3_gmm': 'GMM：用多个高斯分布混合建模；软聚类（概率分配）',
    'fig_5_3_gmm_ellipse': 'GMM用多个高斯混合拟合数据分布，协方差椭圆刻画各簇的形状和方向',
    'fig_5_3_dbscan': 'DBSCAN：基于密度，无需预设K，可处理噪声和任意形状',
    'fig_5_3_4_gmm_ellipse': 'GMM用多个高斯混合拟合数据分布，协方差椭圆刻画各簇的形状和方向',
    'fig_5_3_5_dbscan': '基于密度的聚类：核心点、边界点、噪声点，无需预设簇数，可发现任意形状',
    'fig_5_3_hierarchical': '层次聚类：自底向上合并或自顶向下分裂；树状图',
    'fig_5_3_1_kmeans_iter_detail': 'K-means：①随机初始化中心②分配样本到最近中心③更新中心④迭代收敛',
    'fig_5_3_2_hierarchical_dendrogram': '层次聚类树状图：纵轴是距离/相似度；不同切割得不同簇数',
    'fig_5_3_3_silhouette_k': '轮廓系数s=(b-a)/max(a,b)；a=簇内距离，b=最近簇距离；s越大越好',
    'fig_kmeans_iteration': 'K-means迭代过程：中心移动→簇重新分配→直至收敛或达到最大迭代次数',
    'fig_5_3_cn': '聚类算法对比：K-means、GMM、DBSCAN、层次聚类',
    'fig_5_4_pca': 'PCA：找方差最大方向；主成分是原变量的线性组合',
    'fig_5_4_pca_2d': 'PCA降维到2D可视化；第一主成分解释最大方差',
    'fig_5_4_svd': 'SVD降维：保留最大的k个奇异值对应的特征向量',
    'fig_5_4_1_pca_full_process': 'PCA流程：标准化→求协方差矩阵→求特征值特征向量→按方差贡献率排序→投影',
    'fig_5_4_2_pca_svd_compare': 'PCA的基向量=数据矩阵的右奇异向量；SVD是PCA的推广',
    'fig_5_4_2_dropout': '训练时随机丢弃神经元（概率p）；测试时用完整网络；防止协同适应',
    'fig_5_4_1_backprop_detail': '反向传播：损失函数对每层权重求偏导；链式法则从输出层向输入层反向计算',
    'fig_neural_network_structure': '多层感知机：输入层→隐藏层（可多层）→输出层；激活函数引入非线性',
    'fig_5_4_cn': '主成分分析（PCA）与奇异值分解（SVD）降维',
    'fig_5_5_bagging': 'Bagging：Bootstrap采样+并行学习器+投票/平均',
    'fig_5_5_boosting': 'Boosting：串行聚焦难样本；AdaBoost/RidgeBoost',
    'fig_5_5_xgboost': 'XGBoost：梯度提升的并行化改进；正则化剪枝',
    'fig_5_5_adaboost': 'AdaBoost：每轮提高错分样本权重；指数增长',
    'fig_5_5_stacking': 'Stacking：多层堆叠；Level 0基学习器→Level 1元学习器',
    'fig_5_5_1_bagging_boosting': 'Bagging（并行）：Bootstrap采样+独立学习+投票；Boosting（串行）：聚焦难样本',
    'fig_5_5_2_xgboost_tree': 'XGBoost每棵树学习前面所有树的残差（梯度提升）；正则化项防止过拟合',
    'fig_5_5_3_adaboost_weight': 'AdaBoost每轮增加被错分样本的权重（指数增长），弱分类器加权投票',
    'fig_5_5_4': '随机森林通过特征子采样增加基学习器多样性；OOB估计泛化误差',
    'fig_5_5_5_ensemble_strategies': '集成学习三策略：Bagging降方差、Boosting降偏差、Stacking融合异构模型',
    'fig_ensemble_strategies': '集成学习三策略：Bagging降方差、Boosting降偏差、Stacking融合异构模型',
    'fig_5_5_5_stacking': '两层结构：Level 0多个基学习器输出作为Level 1元学习器的特征',
    'fig_5_5_cn': '集成学习：Bagging（并行）vs Boosting（串行）',
    'fig_5_5_ensemble': '集成学习：组合多个模型提升整体预测性能',
    'fig_5_5_rf': '随机森林：Bagging+决策树；特征子采样增加多样性',
    'fig_5_6_deep': '深度学习：多层神经网络；自动特征学习',
    'fig_5_6_backprop': '反向传播：链式法则求梯度；随机梯度下降训练',
    'fig_5_6_cn': '深度学习基础：前馈网络、激活函数、反向传播',
    'fig_5_ml_overview': '机器学习分类框架：监督/无监督/强化；各方法适用场景',
    'fig_6_1_gd': '梯度下降：沿负梯度方向迭代；学习率决定步长',
    'fig_6_1_newton': '牛顿法：用二阶导数（Hessian）加速收敛；平方级',
    'fig_6_1_1_convex_vs_nonconvex': '凸函数：全局最优唯一；非凸函数有多个局部最优；初始化很重要',
    'fig_6_1_2_gd_vs_newton': '梯度下降一阶收敛；牛顿法二阶收敛（用Hessian）；牛顿法更快但计算量大',
    'fig_gradient_descent': '梯度下降沿负梯度方向迭代；学习率过大震荡，过小收敛慢',
    'fig_6_1_cn': '梯度下降与牛顿法迭代路径对比',
    'fig_6_2_lp': '线性规划：目标函数线性、约束线性；可行域为多面体',
    'fig_6_2_simplex': '单纯形法：沿顶点遍历可行域；指数时间但实际快',
    'fig_6_2_dual': '原问题与对偶问题；弱对偶定理；互补松弛',
    'fig_6_2_1_simplex_path': '单纯形法从可行域一个顶点沿边走到另一个顶点，目标函数值单调增加',
    'fig_6_2_2_lp_dual_shadow': '原问题与对偶问题；弱对偶：原问题目标值≥对偶目标值；影子价格=对偶变量',
    'fig_6_2_cn': '线性规划：可行域、目标函数、等高线、最优解',
    'fig_6_3_ip': '整数规划：部分或全部变量为整数；NP难',
    'fig_6_3_branch_bound': '分支定界：LP松弛下界+分支剪枝；避免枚举',
    'fig_6_3_1_branch_bound': '分支定界通过LP松弛下界和变量分支剪枝，避免枚举所有整数组合',
    'fig_6_3_2_cover_tsp': '集合覆盖用贪心近似（1.58倍近似比）；TSP用最近邻/Christofides启发式',
    'fig_6_3_cn': '整数规划问题及其近似算法',
    'fig_6_4_nonconvex': '非凸优化：多个局部最优；需要随机化或全局搜索',
    'fig_6_4_sa': '模拟退火：以概率接受差解；逐渐降温跳出局部最优',
    'fig_6_4_1_simulated_annealing': 'SA以概率exp(-ΔE/T)接受劣解；温度T随时间下降；理论上可找到全局最优',
    'fig_6_4_cn': '非凸优化：局部最优与全局最优的差距',
    'fig_7_1_ts': '时间序列：趋势、季节性、周期、噪声四要素',
    'fig_7_1_arima': 'ARIMA(p,d,q)：差分+AR+MA；Box-Jenkins方法',
    'fig_7_1_arima_structure': 'AR(p)自回归+I差分+MA(q)移动平均，适用于平稳和非平稳时间序列',
    'fig_7_1_1_fourier_decompose': '傅里叶变换：信号分解为不同频率的正弦波叠加；频谱显示各频率成分强度',
    'fig_7_1_2_smoothing_methods': '移动平均、指数平滑等平滑方法消除噪声、揭示趋势和季节性',
    'fig_7_1_3_differencing': '差分：Δy_t=y_t-y_{t-1}；一阶差分消除趋势；二阶差分消除二次趋势',
    'fig_7_1_4_arima_structure': 'ARIMA(p,d,q)：d阶差分使非平稳序列平稳；AR(p)用前p步自相关；MA(q)用前q步噪声',
    'fig_7_1_acf_pacf': 'ACF衡量直接相关性，PACF衡量剔除中间步影响后的直接相关性',
    'fig_7_1_stationary': '平稳性：均值和方差不随时间变化；ADF检验',
    'fig_timeseries_decomposition': '加法模型Y=T+S+R；乘法模型Y=T·S·R；STL分解灵活处理趋势和季节性',
    'fig_7_1_cn': '时间序列的构成要素：趋势、季节性、周期、噪声',
    'fig_7_2_fft': 'FFT快速傅里叶变换：时域→频域；频谱分析',
    'fig_7_2_filter': '滤波器：低通/高通/带通；FIR和IIR滤波器设计',
    'fig_7_2_1_filter_types': '低通：保留低频；高通：保留高频；带通：保留某个频段；FIR/IIR两类',
    'fig_7_2_spectrogram': '语谱图：STFT时频二维表示；共振峰清晰可见',
    'fig_7_2_2_spectrogram': 'STFT时频图：横轴时间、纵轴频率、颜色深浅表示功率；语谱图揭示共振峰',
    'fig_7_2_3_wavelet': '小波变换用多尺度伸缩窗匹配信号不同频率成分，对非平稳信号比STFT更灵活',
    'fig_7_2_cn': '信号处理：FFT频谱分析、滤波器设计',
    'fig_7_3_audio': '音频特征：过零率、能量、频谱质心、梅尔频谱',
    'fig_7_3_mfcc': 'MFCC：预加重→分帧→FFT→Mel滤波器→对数→DCT',
    'fig_7_3_mfcc_bank': 'Mel滤波器组模拟人耳听觉非线性特性；DCT提取倒谱系数，保留语音识别关键信息',
    'fig_7_3_1_audio_features': '音频特征：过零率（清音/浊音）、频谱质心（音色亮暗）、MFCC（语音识别）',
    'fig_7_3_2_mfcc_bank': 'Mel滤波器组模拟人耳听觉非线性特性；DCT提取倒谱系数，保留语音识别关键信息',
    'fig_7_3_cn': '音频处理：MFCC特征提取流程及各步意义',
    'fig_7_4_music': '音乐挖掘：节拍检测、旋律提取、和弦识别',
    'fig_7_4_1_music_mining': '音乐挖掘：节拍跟踪（节奏感）、旋律提取（主旋律线）、和弦识别（和声结构）',
    'fig_7_4_cn': '音乐挖掘：节拍跟踪、旋律提取、和弦识别、调性分析',
    'fig_acf_pacf': 'ACF衡量直接相关性，PACF衡量剔除中间步影响后的直接相关性',
    'fig_lp_feasible_region': '线性约束在平面上围成的多边形可行域，最优解在顶点',
    'fig_pca_svd': 'PCA与SVD的关系：主成分等于数据矩阵的奇异向量',
    'fig_ml_concepts': '机器学习核心概念：偏差-方差分解、过拟合-欠拟合、交叉验证',
    'fig_classification_metrics': '分类评估指标：准确率、精确率、召回率、F1、AUC-ROC',
    'fig_ensemble_neural': '集成方法与神经网络的演变关系',
    'fig_timeseries_optimization': '时间序列分解与优化算法收敛路径',
}

def get_title_desc(fname):
    key = fname.replace('.png','')
    # Try longer (more specific) keys first
    title = None
    sorted_keys = sorted(TITLE_MAP.keys(), key=len, reverse=True)
    for k in sorted_keys:
        if fname.startswith(k) or key == k:
            title = TITLE_MAP[k]
            break
    if title is None:
        title = key.replace('_',' ')
    desc = DESC_MAP.get(key, DESC_MAP.get(fname.replace('_ai.png',''), title))
    return title, desc


def parse_section_from_path(filepath):
    """Extract (chapter, section) from path like ChapterX_Name/N.M.md"""
    fname = filepath.split('/')[-1]
    digits = re.findall(r'\d+', fname)
    if len(digits) >= 2:
        return digits[0], digits[1]
    return None, None

def process_file(md_relpath, content):
    """Return updated content"""
    chapter, section = parse_section_from_path(md_relpath)
    if chapter is None:
        return content
    
    lines = content.split('\n')
    new_lines = []
    fig_seq = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是图片行：![...](...fig_xxx.png)
        img_m = re.match(r'(!\[[^\]]*\]\([^)]+(fig_[^)]+\.png)\))', line)
        if img_m:
            md_link = img_m.group(1)
            fname = img_m.group(2)
            fig_seq += 1
            fig_num = f"图{chapter}.{section}.{fig_seq}"
            title, desc = get_title_desc(fname)
            # 替换标题（在 [] 中）与链接
            new_link = md_link  # 保持链接不变，只更新 alt text
            # 实际上 alt text 可能旧了，用标准标题替换
            alt_m = re.search(r'!\[([^\]]*)\]', md_link)
            if alt_m:
                old_alt = alt_m.group(1)
                new_link = md_link.replace(f'![{old_alt}]', f'![{title}]')
            
            caption = f"{fig_num} {title}。{desc}"
            new_lines.append(new_link)
            new_lines.append(f"*{caption}*")
            i += 1
            continue
        
        # 检查是否是旧的caption行（*图X.Y.Z 开头），跳过（会在下次图片时重建）
        # 但要处理那些没有对应图片行的旧caption（比如重复插入的）
        cap_m = re.match(r'\*\s*(图\d+\.\d+\.\d+ [^\n]+)\*\s*$', line)
        if cap_m:
            # 跳过旧caption（会在下次遇到图片时重建）
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    result = '\n'.join(new_lines)
    # 清理多余空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

# 遍历
total = 0
md_files = []
for root, dirs, files in os.walk(BASE):
    for fn in sorted(files):
        if fn.endswith('.md'):
            md_files.append(os.path.join(root, fn))

for fpath in sorted(md_files):
    rel = os.path.relpath(fpath, BASE)
    with open(fpath, encoding='utf-8') as f:
        content = f.read()
    
    # 快速检查是否含有图片
    if 'fig_' not in content:
        continue
    
    new_content = process_file(rel, content)
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        # 数一下图数
        fig_count = len(re.findall(r'!\[[^\]]*\]\([^)]+fig_[^)]+\)', new_content))
        print(f"✓ {rel}: {fig_count} figs")
        total += fig_count

print(f"\n总计更新 {total} 张图的编号和说明")
