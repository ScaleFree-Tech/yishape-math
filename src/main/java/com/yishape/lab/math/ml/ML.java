package com.yishape.lab.math.ml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.cls.IClassification;
import com.yishape.lab.math.ml.cls.RereLogisticRegression;
import com.yishape.lab.math.ml.cls.knn.RereKnn;
import com.yishape.lab.math.ml.cls.tree.RereRandomForest;
import com.yishape.lab.math.ml.cls.tree.RereXGboost;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.metric.CrossValidation;
import com.yishape.lab.math.ml.metric.CrossValidationResult;

/**
 * 算法工具类，提供各种机器学习算法的工厂方法和评估工具。
 *
 * 该类作为机器学习算法的统一入口，提供了以下功能：
 * <ul>
 * <li>分类算法的工厂方法</li>
 * <li>分类性能评估工具</li>
 * <li>交叉验证工具</li>
 * </ul>
 *
 * <p>
 * 使用示例：</p>
 * <pre>{@code
 * // 创建逻辑回归分类器
 * IClassification lr = ML.logisticRegression(0.1, 0.01);
 *
 * // 创建随机森林分类器
 * IClassification rf = ML.randomForest();
 *
 * // 计算分类性能指标
 * ClassificationMetrics metrics = ML.classificationMetric(model, features, labels);
 *
 * // 执行k折交叉验证
 * CrossValidationResult cvResult = ML.kFoldCrossValidation(classifier, X, y, 5);
 * }</pre>
 *
 * @author lteb2
 * @since 1.0
 */
public class ML {

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
    public static IClassification logisticRegression(double l1Weight, double l2Weight) {
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
    public static IClassification randomForest() {
        return new RereRandomForest();
    }

    /**
     * 创建XGBoost分类器。
     *
     * <p>
     * XGBoost（Extreme Gradient Boosting）是一种高效的梯度提升框架，具有以下特点：</p>
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
    public static IClassification xGboost() {
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
    public static IClassification kNN(int k) {
        return new RereKnn(k);
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
    public static ClassificationMetrics classificationMetrics(IClassification model, IMatrix feature, String[] trueLabels) {
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
    public static CrossValidationResult kFoldCrossValidation(IClassification classifier,
            IMatrix<Double> X, String[] y, int k) {
        var result = CrossValidation.kFoldCrossValidation(classifier, X, y, k);
        return result;
    }

    /**
     * 从本地保存的分类器模型文件中加载恢复
     *
     * @param modelPath
     * @return
     */
    public static IClassification loadClassifier(String modelPath) {
        try {
            ISerializableModel model = ISerializableModel.load(modelPath);
            return (IClassification) model;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
    
    /**
     * 保存分类器到指定的本地地址
     * @param classifier
     * @param modelPath 
     */
    public static void saveClassifier(IClassification classifier, String modelPath) {
        classifier.save(modelPath);
    }
    
}
