package com.yishape.lab.math.ml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.cls.EnsembleClassifier;
import com.yishape.lab.math.ml.cls.IClassifier;
import com.yishape.lab.math.ml.cls.RereLogisticRegression;
import com.yishape.lab.math.ml.cls.knn.RereKnn;
import com.yishape.lab.math.ml.cls.tree.DecisionTreeCriterion;
import com.yishape.lab.math.ml.cls.tree.RereDecisionTree;
import com.yishape.lab.math.ml.cls.tree.RereRandomForest;
import com.yishape.lab.math.ml.cls.tree.RereXGboost;
import com.yishape.lab.math.ml.cls.svm.RereLinearSVM;
import com.yishape.lab.math.ml.dimreduce.IDimReduce;
import com.yishape.lab.math.ml.dimreduce.RerePCA;
import com.yishape.lab.math.ml.dimreduce.RereSVD;
import com.yishape.lab.math.ml.dimreduce.RereTSNE;
import com.yishape.lab.math.ml.dimreduce.RereUMAP;
import com.yishape.lab.math.ml.lr.IRegression;
import com.yishape.lab.math.ml.lr.RereLinearRegression;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.metric.CrossValidation;
import com.yishape.lab.math.ml.metric.CrossValidationLogger;
import com.yishape.lab.math.ml.metric.CrossValidationResult;

/**
 * 算法工具类，提供各种机器学习算法的工厂方法和评估工具。
 *
 * 该类作为机器学习算法的统一入口，提供了以下功能：
 * <ul>
 * <li>分类算法工厂：{@link #logisticRegression()}、{@link #logisticRegression(double, double)}、
 * {@link #randomForest()}、{@link #decisionTree()}、{@link #linearSvm()}、{@link #xGboost()}、{@link #kNN(int)}、{@link #ensembleClassifier(EnsembleClassifier.EnsembleStrategy, long)} 等</li>
 * <li>线性回归 {@link #linearRegression()} 及正则化快捷构造</li>
 * <li>降维：{@link #pcaDimReducer()}、{@link #svdDimReducer()}、{@link #tsneDimReducer()}、{@link #umapDimReducer()}</li>
 * <li>分类性能评估工具</li>
 * <li>交叉验证工具</li>
 * </ul>
 *
 * <p>
 * 使用示例：</p>
 * <pre>{@code
 * IClassifier lr0 = ML.logisticRegression();
 * IClassifier lr = ML.logisticRegression(0.1, 0.01);
 * IClassifier rf = ML.randomForest();
 * ClassificationMetrics metrics = ML.classificationMetrics(model, features, labels);
 * CrossValidationResult cvResult = ML.kFoldCrossValidation(classifier, X, y, 5);
 * IRegression reg = ML.linearRegression(0.0, 0.1);
 * IMatrix Z = ML.pcaDimReducer().dimensionReduction(X, 2);
 * }</pre>
 *
 * @author lteb2
 * @since 1.0
 */
public class ML {

    private static final Logger log = LoggerFactory.getLogger(ML.class);


    /**
     * 创建逻辑回归分类器（无 L1/L2 正则，等价于 {@code new RereLogisticRegression()}）。
     *
     * @return 默认配置的逻辑回归实例
     * @see RereLogisticRegression#RereLogisticRegression()
     */
    public static IClassifier logisticRegression() {
        return new RereLogisticRegression();
    }

    /**
     * 创建带有L1和L2正则化的逻辑回归分类器。
     *
     * <p>
     * 逻辑回归是一种广泛使用的线性分类算法，通过添加正则化项可以防止过拟合。
     * L1正则化（Lasso）可以产生稀疏解，有助于特征选择；L2正则化（Ridge）可以防止权重过大。</p>
     *
     * @param l1Weight L1正则化权重系数，控制L1正则化的强度
     * @param l2Weight L2正则化权重系数，控制L2正则化的强度
     * @return 配置好的逻辑回归分类器实例
     * @throws IllegalArgumentException 当权重系数为负数时抛出
     *
     * @see RereLogisticRegression
     * @see <a href="https://en.wikipedia.org/wiki/Logistic_regression">Logistic
     * Regression</a>
     */
    public static IClassifier logisticRegression(double l1Weight, double l2Weight) {
        return new RereLogisticRegression(l1Weight, l2Weight);
    }

    /**
     * 创建随机森林分类器。
     *
     * <p>
     * 随机森林是一种集成学习方法，通过构建多个决策树并进行投票来提高分类的准确性和稳定性。 它具有以下优点：</p>
     * <ul>
     * <li>能够处理高维数据</li>
     * <li>对过拟合有很好的抵抗力</li>
     * <li>可以评估特征的重要性</li>
     * <li>不需要特征缩放</li>
     * </ul>
     *
     * @return 配置好的随机森林分类器实例
     *
     * @see RereRandomForest
     * @see <a href="https://en.wikipedia.org/wiki/Random_forest">Random
     * Forest</a>
     */
    public static IClassifier randomForest() {
        return new RereRandomForest();
    }

    /**
     * 创建XGBoost分类器。
     *
     * <p>
     * XGBoost（Extreme Gradient Boosting）是一种高效的梯度提升框架；默认含标签频率估计的初始 margin（{@code base_score}）
     * 与 Hessian 加权分位数直方图，可通过 {@link RereXGboost#setHistogramBinning} / {@link RereXGboost#setBaseMargin} 调整。</p>
     * <ul>
     * <li>高预测精度</li>
     * <li>快速的训练速度</li>
     * <li>内置的正则化功能</li>
     * <li>支持并行和分布式计算</li>
     * <li>能够处理缺失值</li>
     * </ul>
     *
     * @return 配置好的XGBoost分类器实例
     *
     * @see RereXGboost
     * @see <a href="https://xgboost.readthedocs.io/">XGBoost Documentation</a>
     */
    public static IClassifier xGboost() {
        return new RereXGboost();
    }

    /**
     * 创建K近邻（K-Nearest Neighbors）分类器。
     *
     * <p>
     * KNN是一种基于实例的学习算法，通过查找训练集中与新样本最相似的k个邻居来进行分类。 算法的工作原理：</p>
     * <ol>
     * <li>计算新样本与训练集中所有样本的距离</li>
     * <li>选择距离最近的k个样本</li>
     * <li>根据这k个邻居的类别进行投票，得票最多的类别作为预测结果</li>
     * </ol>
     *
     * @param k 邻居数量，必须是正整数
     * @return 配置好的KNN分类器实例
     * @throws IllegalArgumentException 当k小于等于0时抛出
     *
     * @see RereKnn
     * @see
     * <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">K-Nearest
     * Neighbors</a>
     */
    public static IClassifier kNN(int k) {
        return new RereKnn(k);
    }

    /**
     * 创建集成分类器（投票 / 加权投票 / 堆叠），等价于 {@code new EnsembleClassifier(strategy, randomSeed)}。
     *
     * @param strategy   集成策略
     * @param randomSeed 随机种子
     * @return 集成分类器实例
     * @see EnsembleClassifier
     */
    public static IClassifier ensembleClassifier(EnsembleClassifier.EnsembleStrategy strategy, long randomSeed) {
        return new EnsembleClassifier(strategy, randomSeed);
    }

    /**
     * 创建带权重向量的集成分类器（用于加权投票等），等价于
     * {@code new EnsembleClassifier(strategy, weights, randomSeed)}。
     *
     * @param strategy   集成策略
     * @param weights    各基分类器权重
     * @param randomSeed 随机种子
     * @return 集成分类器实例
     */
    public static IClassifier ensembleClassifier(EnsembleClassifier.EnsembleStrategy strategy,
            double[] weights, long randomSeed) {
        return new EnsembleClassifier(strategy, weights, randomSeed);
    }

    /**
     * 单棵决策树（默认 CART 基尼不纯度、深度不限由样本划分终止条件约束）。
     *
     * @see RereDecisionTree
     * @see DecisionTreeCriterion
     */
    public static IClassifier decisionTree() {
        return new RereDecisionTree();
    }

    /**
     * 单棵决策树（指定准则与深度 / 分裂约束）。
     *
     * @param criterion        {@link DecisionTreeCriterion#CART_GINI} / {@link DecisionTreeCriterion#CART_ENTROPY} /
     *                         {@link DecisionTreeCriterion#C45_GAIN_RATIO}
     * @param maxDepth         最大深度（≤0 表示不限制）
     * @param minSamplesSplit  结点至少样本数才允许分裂
     * @param minSamplesLeaf   叶结点最少样本数
     */
    public static IClassifier decisionTree(DecisionTreeCriterion criterion, int maxDepth,
            int minSamplesSplit, int minSamplesLeaf) {
        return new RereDecisionTree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf);
    }

    /**
     * 线性 SVM（平方铰链 + L2，OvR 多分类），默认 {@code C=1}、训练集标准化。
     *
     * @see RereLinearSVM
     */
    public static IClassifier linearSvm() {
        return new RereLinearSVM();
    }

    /**
     * 线性 SVM，指定铰链损失权重 {@code C} 与是否标准化特征。
     */
    public static IClassifier linearSvm(double C, boolean standardizeFeatures) {
        return new RereLinearSVM(C, standardizeFeatures);
    }

    // ========== 回归 / Regression ==========

    /**
     * 普通最小二乘线性回归（含截距、无正则化），等价于 {@code new RereLinearRegression()}。
     *
     * @return 
     * @see RereLinearRegression
     */
    public static IRegression linearRegression() {
        return new RereLinearRegression();
    }

    /**
     * 线性回归（含截距），根据 {@code lambda1}、{@code lambda2} 自动选择无正则 / L1 / L2 / ElasticNet，
     * 等价于 {@code new RereLinearRegression(true, lambda1, lambda2)}。
     *
     * @param lambda1 L1 系数（Lasso）
     * @param lambda2 L2 系数（Ridge）
     * @return 
     * @see RereLinearRegression
     */
    public static IRegression linearRegression(double lambda1, double lambda2) {
        return new RereLinearRegression(true, lambda1, lambda2);
    }

    /**
     * 线性回归（可关闭截距），正则类型由 {@code lambda1}、{@code lambda2} 推断。
     *
     * @param includeBias 是否包含偏置项
     * @param lambda1     L1 系数
     * @param lambda2     L2 系数
     * @return 
     * @see RereLinearRegression
     */
    public static IRegression linearRegression(boolean includeBias, double lambda1, double lambda2) {
        return new RereLinearRegression(includeBias, lambda1, lambda2);
    }



    // ========== 降维 / Dimensionality reduction ==========

    /** PCA 降维器实例，等价于 {@code new RerePCA()}。
     * @return  */
    public static IDimReduce pcaDimReducer() {
        return new RerePCA();
    }

    /** 基于 SVD 的线性降维器，等价于 {@code new RereSVD()}。
     * @return  */
    public static IDimReduce svdDimReducer() {
        return new RereSVD();
    }

    /** t-SNE 非线性降维器，等价于 {@code new RereTSNE()}。
     * @return  */
    public static IDimReduce tsneDimReducer() {
        return new RereTSNE();
    }

    /** UMAP 非线性降维器，等价于 {@code new RereUMAP()}。
     * @return  */
    public static IDimReduce umapDimReducer() {
        return new RereUMAP();
    }


    /**
     * 计算分类器的性能指标。
     *
     * <p>
     * 该方法使用真实标签和分类器的预测结果来计算各种分类性能指标，包括：</p>
     * <ul>
     * <li>准确率（Accuracy）</li>
     * <li>精确率（Precision）</li>
     * <li>召回率（Recall）</li>
     * <li>F1分数（F1-Score）</li>
     * <li>混淆矩阵（Confusion Matrix）</li>
     * </ul>
     *
     * @param model 训练好的分类器模型
     * @param feature 输入特征矩阵，每一行代表一个样本，每一列代表一个特征
     * @param trueLabels 真实的类别标签数组，长度必须与特征矩阵的行数一致
     * @return 包含各种分类性能指标的结果对象
     * @throws IllegalArgumentException 当特征矩阵与标签数组维度不匹配时抛出
     *
     * @see ClassificationMetrics
     * @see
     * <a href="https://en.wikipedia.org/wiki/Precision_and_recall">Precision
     * and Recall</a>
     */
    public static ClassificationMetrics classificationMetrics(IClassifier model, IMatrix feature, String[] trueLabels) {
        ClassificationMetrics metrics = ClassificationMetrics.compute(model, feature, trueLabels);
        return metrics;
    }

    /**
     * 执行k折交叉验证来评估分类器的性能。
     *
     * <p>
     * 交叉验证是一种评估模型泛化能力的统计方法。k折交叉验证的步骤：</p>
     * <ol>
     * <li>将数据集随机分成k个大小相等的子集</li>
     * <li>对于每个子集，使用其余k-1个子集作为训练集，该子集作为测试集</li>
     * <li>训练模型并在测试集上评估性能</li>
     * <li>重复k次，计算平均性能指标</li>
     * </ol>
     *
     * @param classifier 要评估的分类器
     * @param X 特征矩阵，每一行代表一个样本，每一列代表一个特征
     * @param y 类别标签数组，长度必须与特征矩阵的行数一致
     * @param k 折数，通常取5或10
     * @return 交叉验证结果，包含每次验证的性能指标和平均值
     * @throws IllegalArgumentException 当k小于2或数据集大小不足以进行k折分割时抛出
     *
     * @see CrossValidation
     * @see CrossValidationResult
     * @see
     * <a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)">Cross-Validation</a>
     */
    public static CrossValidationResult kFoldCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y, int k) {
        var result = CrossValidation.kFoldCrossValidation(classifier, X, y, k);
        return result;
    }
    
        /**
     * 执行k折交叉验证来评估分类器的性能。
     *
     * <p>
     * 交叉验证是一种评估模型泛化能力的统计方法。k折交叉验证的步骤：</p>
     * <ol>
     * <li>将数据集随机分成k个大小相等的子集</li>
     * <li>对于每个子集，使用其余k-1个子集作为训练集，该子集作为测试集</li>
     * <li>训练模型并在测试集上评估性能</li>
     * <li>重复k次，计算平均性能指标</li>
     * </ol>
     *
     * @param classifier 要评估的分类器
     * @param X 特征矩阵，每一行代表一个样本，每一列代表一个特征
     * @param y 类别标签数组，长度必须与特征矩阵的行数一致
     * @param k 折数，通常取5或10
     * @param logger 交叉检验折间输出的训练日志
     * @return 交叉验证结果，包含每次验证的性能指标和平均值
     * @throws IllegalArgumentException 当k小于2或数据集大小不足以进行k折分割时抛出
     *
     * @see CrossValidation
     * @see CrossValidationResult
     * @see
     * <a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)">Cross-Validation</a>
     */
    public static CrossValidationResult kFoldCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y, int k, CrossValidationLogger logger) {
        var result = CrossValidation.kFoldCrossValidation(classifier, X, y, k,logger);
        return result;
    }

    /**
     * 从本地保存的分类器模型文件中加载恢复
     *
     * @param modelPath 模型文件路径 / Model file path
     * @return 加载的分类器 / Loaded classifier
     * @throws IllegalStateException 如果加载失败 / if loading fails
     */
    public static IClassifier loadClassifier(String modelPath) {
        try {
            ISerializableModel model = ISerializableModel.load(modelPath);
            return (IClassifier) model;
        } catch (Exception e) {
            log.error("Failed to load classifier from: {}", modelPath, e);
            throw new IllegalStateException("Failed to load classifier from: " + modelPath, e);
        }
    }
    
    /**
     * 保存分类器到指定的本地地址
     * @param classifier
     * @param modelPath 
     */
    public static void saveClassifier(IClassifier classifier, String modelPath) {
        classifier.save(modelPath);
    }
    
}
