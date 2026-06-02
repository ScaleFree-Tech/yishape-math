package com.yishape.lab.math.ml.reg;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.regularization.RereL1Regularization;
import com.yishape.lab.math.optimize.regularization.RereL2Regularization;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.autodiff.AD;

import java.util.LinkedHashMap;
import java.util.Map;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * 线性回归实现类
 * <p>
 * 本类实现了标准的线性回归算法，使用最小二乘法优化目标函数。
 * 线性回归模型形式：y = w^T * x + b，其中：
 * - w 是权重向量（特征系数）
 * - x 是输入特征向量
 * - b 是偏置项（截距）
 * - y 是预测值
 * </p>
 * <p>
 * 算法特点：
 * 1. 使用LBFGS优化器求解最优权重
 * 2. 支持带偏置项和不带偏置项的线性回归
 * 3. 目标函数：均方误差损失 + 正则化项（L1、L2或ElasticNet；L1/L2 惩罚与梯度由
 * {@link RereL1Regularization} / {@link RereL2Regularization} 计算，见各类说明）
 * 4. 自动处理特征矩阵的增广（添加偏置列）
 * </p>
 * 
* <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建线性回归模型
 RereLinearRegression lr = new RereLinearRegression();

 // 训练模型（支持链式调用）
 lr.fit(featureMatrix, labelVector);

 // 或获取结果对象
 RegressionResult result = lr.getResult();
 double loss = result.getLoss();

 // 预测新样本
 double prediction = lr.predict(newFeatureVector);

 // 获取模型权重
 IVector weights = lr.getFeatureWeights();
 }
 * </pre>
 * 
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class RereLinearRegression implements IRegression, IGradientFunction, IObjectiveFunction {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereLinearRegression.class);

    /**
     * 正则化类型枚举
     */
    public enum RegularizationType {
        /** 无正则化 */
        NONE,
        /** L1正则化（Lasso） */
        L1,
        /** L2正则化（Ridge） */
        L2,
        /** ElasticNet正则化（L1 + L2的组合） */
        ELASTIC_NET
    }
    
    /**
     * 优化器，用于求解最优权重
     */
    private IOptimizer optimizer = new RereLBFGS();

    /**
     * 批量优化器（默认 L-BFGS）的迭代上界。凸带正则最小二乘通常远早于此即因梯度范数达标而停止。
     */
    private int maxOptimizerIterations = 500;

    /**
     * 批量优化器收敛容差（梯度范数阈值）。
     */
    private double optimizerTolerance = 1e-6;
    
    /**
     * 训练后的权重向量（包含偏置项）
     */
    private IVector trainedWeights = null;

    /**
     * 训练结果对象
     */
    private RegressionResult result;
    
    /**
     * 是否包含偏置项
     */
    private boolean includeBias = true;
    
    /**
     * 正则化类型
     */
    private RegularizationType regularizationType = RegularizationType.NONE;
    
    /**
     * L1正则化系数（λ₁）
     */
    private double lambda1 = 0.0;
    
    /**
     * L2正则化系数（λ₂）
     */
    private double lambda2 = 0.0;

    /**
     * L1 惩罚：Huber 光滑近似（与 DDML 参考实现一致），非 {@code λ₁‖w‖₁} 次梯度形式。
     */
    private final RereL1Regularization l1Regularization = new RereL1Regularization();

    /**
     * L2 惩罚：值为 {@link RereL2Regularization#computeObjective} = ‖w‖²，本类目标中为 (λ₂/2)‖w‖²。
     */
    private final RereL2Regularization l2Regularization = new RereL2Regularization();

    /**
     * 训练特征矩阵（增广后的）
     */
    private IMatrix augmentedFeatures = null;
    
    /**
     * 训练标签向量
     */
    private IVector trainingLabels = null;
    
    /**
     * 特征数量（不包括偏置项）
     */
    private int featureCount = 0;
    
    /**
     * 样本数量
     */
    private int sampleCount = 0;

    /**
     * 默认构造函数
     * <p>
     * 使用默认参数创建线性回归模型：
     * - 包含偏置项
     * - 无正则化
     * - 使用LBFGS优化器
     * </p>
     */
    public RereLinearRegression() {
        this(true, 0.0, 0.0);
    }
    
    /**
     * 构造函数（向后兼容）
     * 
     * @param includeBias 是否包含偏置项
     * @param lambda L2正则化系数（已废弃，建议使用新的构造函数）
     * @deprecated 建议使用 {@link #RereLinearRegression(boolean, double, double)}
     *             此构造函数将lambda参数用作L2正则化系数，L1正则化系数设置为0.0
     */
    @Deprecated
    public RereLinearRegression(boolean includeBias, double lambda) {
        this(includeBias, 0.0, lambda);
    }
    
    /**
     * 构造函数（推荐使用）
     * 
     * @param includeBias 是否包含偏置项
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     */
    public RereLinearRegression(boolean includeBias, double lambda1, double lambda2) {
        this.includeBias = includeBias;
        updateRegularizationType(lambda1, lambda2);
    }
    
    /**
     * 构造函数
     * 
     * @param includeBias 是否包含偏置项
     * @param regularizationType 正则化类型
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     */
    public RereLinearRegression(boolean includeBias, RegularizationType regularizationType, double lambda1, double lambda2) {
        this.includeBias = includeBias;
        this.regularizationType = regularizationType;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        
        // 验证参数
        validateRegularizationParameters();
    }
    
    /**
     * 构造函数
     * 
     * @param includeBias 是否包含偏置项
     * @param regularizationType 正则化类型
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     * @param optimizer 自定义优化器
     */
    public RereLinearRegression(boolean includeBias, RegularizationType regularizationType, 
                               double lambda1, double lambda2, IOptimizer optimizer) {
        this.includeBias = includeBias;
        this.regularizationType = regularizationType;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.optimizer = optimizer;
        
        // 验证参数
        validateRegularizationParameters();
    }
    
    /**
     * 验证正则化参数的有效性
     * 
     * @throws IllegalArgumentException 如果正则化参数无效
     */
    private void validateRegularizationParameters() {
        switch (regularizationType) {
            case L1:
                if (lambda1 <= 0) {
                    throw new IllegalArgumentException("L1正则化系数必须大于0");
                }
                break;
            case L2:
                if (lambda2 <= 0) {
                    throw new IllegalArgumentException("L2正则化系数必须大于0");
                }
                break;
            case ELASTIC_NET:
                if (lambda1 <= 0 || lambda2 <= 0) {
                    throw new IllegalArgumentException("ElasticNet正则化系数必须都大于0");
                }
                break;
            case NONE:
                // 无正则化，参数可以为0
                break;
        }
    }
    
    /**
     * 根据lambda1和lambda2的值自动判断正则化类型
     * 
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     * @return 推断出的正则化类型
     */
    private RegularizationType inferRegularizationType(double lambda1, double lambda2) {
        if (lambda1 > 0 && lambda2 > 0) {
            return RegularizationType.ELASTIC_NET;
        } else if (lambda1 > 0 && lambda2 <= 0) {
            return RegularizationType.L1;
        } else if (lambda1 <= 0 && lambda2 > 0) {
            return RegularizationType.L2;
        } else {
            return RegularizationType.NONE;
        }
    }
    
    /**
     * 更新正则化类型并验证参数
     * 
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     */
    private void updateRegularizationType(double lambda1, double lambda2) {
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.regularizationType = inferRegularizationType(lambda1, lambda2);
        validateRegularizationParameters();
    }

    
    /**
     * 训练线性回归模型
     * <p>
     * 训练过程：
     * 1. 数据预处理：增广特征矩阵（添加偏置列）
     * 2. 初始化权重：零向量或随机初始化
     * 3. 使用优化器求解最优权重
     * 4. 计算最终损失值
     * 5. 存储训练结果
     * </p>
     *
     * @param feature 特征矩阵，每行是一个样本，每列是一个特征
     * @param labels 标签向量，对应每个样本的真实值
     * @return 当前实例，支持链式调用
     * @throws IllegalArgumentException 如果输入参数无效
     */
    @Deprecated
    public IRegression fitWithManualGradient(IMatrix feature, IVector labels) {
        // 参数验证
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签向量不能为null");
        }
        if (feature.getRowNum() != labels.length()) {
            throw new IllegalArgumentException("样本数量不匹配：特征矩阵行数(" + feature.getRowNum() +
                ") != 标签向量长度(" + labels.length() + ")");
        }
        if (feature.getRowNum() == 0) {
            throw new IllegalArgumentException("训练数据不能为空");
        }

        // 保存训练数据
        this.sampleCount = feature.getRowNum();
        this.featureCount = feature.getColNum();
        this.trainingLabels = labels;

        // 增广特征矩阵（添加偏置列）
        this.augmentedFeatures = augmentFeatures(feature);

        // 初始化权重向量
        int weightCount = this.augmentedFeatures.getColNum();
        IVector initialWeights = IVector.zeros(weightCount);

        // 同步 L-BFGS 参数（上界仅为安全网，多数训练在远少于该步数内结束）
        applyLbfgsSettingsIfApplicable();
        // 使用优化器求解最优权重
        var optimizationResult = optimizer.optimize(
            initialWeights, this, this);

        // 保存训练结果
        this.trainedWeights = optimizationResult.getOptimalPoint();
        double finalLoss = optimizationResult.getOptimalValue();

        // 创建并存储训练结果
        this.result = new RegressionResult();

        // 分离特征权重和偏置项
        if (this.includeBias && this.trainedWeights != null) {
            // 包含偏置项：权重向量的最后一个元素是偏置项
            int featureWeightCount = this.trainedWeights.length() - 1;

            // 提取特征权重（不包括偏置项）
            IVector featureWeightVector = this.trainedWeights.slice(0, featureWeightCount);

            // 提取偏置项（权重向量的最后一个元素）
            double biasValue = (double)this.trainedWeights.get(featureWeightCount);
            IVector biasVector = IVector.of(new double[]{biasValue});

            result.setWeights(featureWeightVector);
            result.setBias(biasVector);
        } else {
            // 不包含偏置项：整个权重向量都是特征权重
            result.setWeights(this.trainedWeights);
            result.setBias(IVector.of(new double[]{0.0})); // 偏置项为0
        }

        result.setLoss(finalLoss);
        result.setR2Score(computeR2Score(feature, labels));
        result.setRmse(computeRmse(feature, labels));

        return this;
    }

    @Override
    public IRegression fit(IMatrix feature, IVector labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签向量不能为null");
        }
        this.sampleCount = feature.rows();
        this.featureCount = feature.cols();
        this.trainingLabels = labels;
        this.augmentedFeatures = augmentFeatures(feature);

        int weightCount = this.augmentedFeatures.getColNum();
        IDiffMatrix Xc = AD.matrix((IDoubleMatrix) this.augmentedFeatures);
        IDiffVector yc = AD.constant((IDoubleVector) labels);

        IVector w0 = IVector.zeros(weightCount);
        var optResult = AD.optimize(w0, w -> {
            IDiffVector predictions = Xc.matmul(w);
            IDiffVector errors = predictions.sub(yc);
            return errors.square().mean().div(2.0);
        }, this.optimizer);

        this.trainedWeights = optResult.getOptimalPoint();

        this.result = new RegressionResult();
        if (this.includeBias && this.trainedWeights != null) {
            int fwc = this.trainedWeights.length() - 1;
            this.result.setWeights(this.trainedWeights.slice(0, fwc));
            this.result.setBias(IVector.of(new double[]{(double)this.trainedWeights.get(fwc)}));
        } else {
            this.result.setWeights(this.trainedWeights);
            this.result.setBias(IVector.of(new double[]{0.0}));
        }
        this.result.setLoss(optResult.getOptimalValue());
        this.result.setR2Score(computeR2Score(feature, labels));
        this.result.setRmse(computeRmse(feature, labels));
        return this;
    }

    @Override
    public double[] fitPredict(IMatrix feature, IVector labels) {
        fit(feature, labels);
        return predictBatch(feature);
    }

    @Override
    public double[] predictBatch(IMatrix features) {
        double[] predictions = new double[features.getRowNum()];
        for (int i = 0; i < features.getRowNum(); i++) {
            predictions[i] = predict(features.getRow(i));
        }
        return predictions;
    }

    @Override
    /**
     * 基于输入特征向量预测目标值
     * <p>
     * 预测公式：y = w^T * x_aug，其中：
     * - w 是训练得到的权重向量
     * - x_aug 是增广后的特征向量（包含偏置项）
     * </p>
     * 
     * @param x 输入特征向量
     * @return 预测值
     * @throws IllegalStateException 如果模型尚未训练
     * @throws IllegalArgumentException 如果输入特征维度不匹配
     */
    public double predict(IVector x) {
        if (this.trainedWeights == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (x == null) {
            throw new IllegalArgumentException("输入特征向量不能为null");
        }
        if (x.length() != this.featureCount) {
            throw new IllegalArgumentException("特征维度不匹配：输入(" + x.length() + 
                ") != 训练时特征数(" + this.featureCount + ")");
        }
        
        // 增广输入特征向量
        IVector augmentedX = augmentFeatureVector(x);
        
        // 计算预测值：w^T * x
        double prediction = this.trainedWeights.innerProductValue(augmentedX);
        
        return prediction;
    }

    /**
     * 决定系数 R²（{@code 1 - SS_res/SS_tot}），在给定特征与标签上计算。
     * <p>
     * 训练集 R²、RMSE 已在 {@link RegressionResult#getR2Score()}、{@link RegressionResult#getRmse()} 中给出。
     * 本方法用于<strong>验证集、测试集</strong>等另一组 (X, y) 的 R²。
     * </p>
     *
     * @param features 特征矩阵（列数须与训练时一致）
     * @param labels 真实标签
     * @return R²；若标签无方差且残差为 0 返回 1.0，若常数标签但残差非 0 返回 0.0
     * @throws IllegalStateException 若尚未训练
     */
    public double r2ScoreOn(IMatrix features, IVector labels) {
        return computeR2Score(features, labels);
    }

    /**
     * 均方根误差 RMSE（{@code sqrt( (1/n) * Σ(y - ŷ)² )}），在给定特征与标签上计算。
     * <p>
     * 训练集 RMSE 见 {@link RegressionResult#getRmse()}；本方法用于验证集、测试集等。
     * </p>
     *
     * @param features 特征矩阵（列数须与训练时一致）
     * @param labels     真实标签
     * @return RMSE；{@code n == 0} 时不在此返回（由校验抛出）
     */
    public double rmseOn(IMatrix features, IVector labels) {
        return computeRmse(features, labels);
    }

    private double computeR2Score(IMatrix features, IVector labels) {
        if (this.trainedWeights == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (features == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签向量不能为null");
        }
        int n = features.getRowNum();
        if (n != labels.length()) {
            throw new IllegalArgumentException("样本数量不匹配：特征矩阵行数(" + n
                    + ") != 标签向量长度(" + labels.length() + ")");
        }
        if (n == 0) {
            throw new IllegalArgumentException("样本数量不能为0");
        }
        if (features.getColNum() != this.featureCount) {
            throw new IllegalArgumentException("特征维度不匹配：输入列数(" + features.getColNum()
                    + ") != 训练时特征数(" + this.featureCount + ")");
        }
        double meanSum = 0.0;
        for (int i = 0; i < n; i++) {
            meanSum += labels.get(i);
        }
        double yMean = meanSum / n;
        double ssTot = 0.0;
        for (int i = 0; i < n; i++) {
            double d = labels.get(i) - yMean;
            ssTot += d * d;
        }
        double ssRes = 0.0;
        for (int i = 0; i < n; i++) {
            double yHat = predict(features.getRow(i));
            double e = labels.get(i) - yHat;
            ssRes += e * e;
        }
        final double eps = 1e-15;
        if (ssTot < eps) {
            return ssRes < eps ? 1.0 : 0.0;
        }
        return 1.0 - ssRes / ssTot;
    }

    private double computeRmse(IMatrix features, IVector labels) {
        if (this.trainedWeights == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (features == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签向量不能为null");
        }
        int n = features.getRowNum();
        if (n != labels.length()) {
            throw new IllegalArgumentException("样本数量不匹配：特征矩阵行数(" + n
                    + ") != 标签向量长度(" + labels.length() + ")");
        }
        if (n == 0) {
            throw new IllegalArgumentException("样本数量不能为0");
        }
        if (features.getColNum() != this.featureCount) {
            throw new IllegalArgumentException("特征维度不匹配：输入列数(" + features.getColNum()
                    + ") != 训练时特征数(" + this.featureCount + ")");
        }
        double ssRes = 0.0;
        for (int i = 0; i < n; i++) {
            double yHat = predict(features.getRow(i));
            double e = labels.get(i) - yHat;
            ssRes += e * e;
        }
        return Math.sqrt(ssRes / n);
    }

    @Override
    /**
     * 计算目标函数的梯度
     * <p>
     * 对于线性回归，目标函数为：
     * J(w) = (1/2n) * ||X*w - y||^2 + R(w)
     * 
     * 其中 R(w) 为惩罚项（L1 使用 {@link RereL1Regularization} 的 Huber 光滑 L1；L2 为 (λ₂/2)||w||²，由 {@link RereL2Regularization} 与系数组合得到）：
     * - L1：λ₁ ×（光滑 L1 和）
     * - L2：(λ₂/2)||w||²
     * - ElasticNet：上述两者之和
     * 
     * 梯度为：
     * ∇J(w) = (1/n) * X^T * (X*w - y) + ∇R(w)
     * 
     * 其中：
     * - X 是增广后的特征矩阵
     * - w 是权重向量
     * - y 是标签向量
     * - λ₁, λ₂ 是正则化系数
     * - n 是样本数量
     * </p>
     * 
     * @param w 权重向量
     * @return 梯度向量
     */
    public IVector computeGradient(IVector w) {
        if (this.augmentedFeatures == null || this.trainingLabels == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        // 计算预测值：X * w
        IVector predictions = computePredictions(w);
        
        // 计算残差：X * w - y
        IVector residuals = (IVector)predictions.sub(this.trainingLabels);
        
        // 计算梯度：(1/n) * X^T * residuals
        IVector gradient = computeMatrixVectorGradient(residuals);
        
        // 添加正则化项的梯度
        IVector regularizationGradient = computeRegularizationGradient(w);
        if (regularizationGradient != null) {
            gradient = (IVector)gradient.add(regularizationGradient);
        }
        
        return gradient;
    }

    @Override
    /**
     * 计算目标函数值
     * <p>
     * 目标函数：J(w) = (1/2n) * ||X*w - y||^2 + R(w)
     * 
     * 其中：
     * - 第一项是均方误差损失
     * - 第二项是正则化项（L1、L2或ElasticNet）
     * - n 是样本数量
     * </p>
     * 
     * @param w 权重向量
     * @return 目标函数值
     */
    public double computeObjective(IVector w) {
        if (this.augmentedFeatures == null || this.trainingLabels == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        // 计算预测值
        IVector predictions = computePredictions(w);
        
        // 计算残差
        IVector residuals = (IVector)predictions.sub(this.trainingLabels);
        
        // 计算均方误差：(1/2n) * ||residuals||^2
        double mse = residuals.innerProductValue(residuals) / (2.0 * this.sampleCount);
        
        // 计算正则化项
        double regularization = computeRegularizationTerm(w);
        
        return mse + regularization;
    }
    
    /**
     * 计算正则化项的值
     * 
     * @param w 权重向量
     * @return 正则化项的值
     */
    private double computeRegularizationTerm(IVector w) {
        switch (regularizationType) {
            case L1:
                return lambda1 * l1Regularization.computeObjective(w);
            case L2:
                return (lambda2 / 2.0) * l2Regularization.computeObjective(w);
            case ELASTIC_NET:
                return lambda1 * l1Regularization.computeObjective(w)
                        + (lambda2 / 2.0) * l2Regularization.computeObjective(w);
            case NONE:
            default:
                return 0.0f;
        }
    }
    
    /**
     * 计算正则化项的梯度
     * 
     * @param w 权重向量
     * @return 正则化项的梯度向量，如果无正则化则返回null
     */
    private IVector computeRegularizationGradient(IVector w) {
        switch (regularizationType) {
            case L1:
                return (IVector) l1Regularization.computeGradient(w).multiplyByScalar(lambda1);
            case L2:
                return (IVector) l2Regularization.computeGradient(w).multiplyByScalar(lambda2 / 2.0);
            case ELASTIC_NET:
                return computeElasticNetGradient(w);
            case NONE:
            default:
                return null;
        }
    }

    /**
     * ElasticNet：光滑 L1 与 L2 梯度之和；L2 与 (λ₂/2)||w||² 一致故为 (λ₂/2)×2w = λ₂ w。
     */
    private IVector computeElasticNetGradient(IVector w) {
        IVector l1Part = (IVector) l1Regularization.computeGradient(w).multiplyByScalar(lambda1);
        IVector l2Part = (IVector) l2Regularization.computeGradient(w).multiplyByScalar(lambda2 / 2.0);
        return (IVector) l1Part.add(l2Part);
    }
    
    /**
     * 增广特征矩阵，添加偏置列
     * <p>
     * 如果includeBias为true，在特征矩阵右侧添加一列全1的偏置列：
     * [X] -> [X | 1]
     * 
     * 如果includeBias为false，直接返回原特征矩阵
     * </p>
     * 
     * @param feature 原始特征矩阵
     * @return 增广后的特征矩阵
     */
    private IMatrix augmentFeatures(IMatrix feature) {
        if (!this.includeBias) {
            return feature;
        }
        
        int rows = feature.getRowNum();
        
        // 创建一列全1的偏置列向量
        IVector onesVector = IVector.ones(rows);
        
        // 将向量转换为列矩阵
        IMatrix biasColumn = onesVector.asColumnVector();
        
        // 使用水平连接将偏置列添加到特征矩阵右侧
        return feature.hstack(biasColumn);
    }
    
    /**
     * 增广单个特征向量，添加偏置项
     * <p>
     * 如果includeBias为true，在特征向量末尾添加偏置项1：
     * [x1, x2, ..., xn] -> [x1, x2, ..., xn, 1]
     * 
     * 如果includeBias为false，直接返回原特征向量
     * </p>
     * 
     * @param x 原始特征向量
     * @return 增广后的特征向量
     */
    private IVector augmentFeatureVector(IVector x) {
        if (!this.includeBias) {
            return x;
        }
        
        // 使用向量连接将偏置项添加到特征向量末尾
        // 使用concat方法替代手动数组操作
        IVector biasElement = IVector.of(new double[]{1.0});
        return x.concat(biasElement);
    }
    
    /**
     * 计算预测值：X * w
     * <p>
     * 使用矩阵-向量乘法计算所有样本的预测值
     * </p>
     * 
     * @param w 权重向量
     * @return 预测值向量
     */
    private IVector computePredictions(IVector w) {
        // 使用矩阵-向量乘法计算预测值
        // 现在使用IMatrix接口提供的mmul方法进行矩阵-向量乘法
        return this.augmentedFeatures.mmul(w);
    }
    
    /**
     * 计算梯度中的矩阵-向量部分：(1/n) * X^T * residuals
     * <p>
     * 这是梯度计算的核心部分，计算特征矩阵转置与残差向量的乘积
     * </p>
     * 
     * @param residuals 残差向量
     * @return 梯度向量
     */
    private IVector computeMatrixVectorGradient(IVector residuals) {
        // 计算 (1/n) * X^T * residuals
        // 使用矩阵转置与向量的乘法：先转置矩阵，再与向量相乘
        IMatrix transposedFeatures = this.augmentedFeatures.transpose();
        IVector gradient = transposedFeatures.mmul(residuals);
        
        // 除以样本数量n
        return gradient.multiplyByScalar(1.0 / this.sampleCount);
    }
    
    /**
     * 获取训练后的权重向量
     * 
     * @return 权重向量，如果未训练则返回null
     */
    public IVector getWeights() {
        return this.trainedWeights;
    }
    
    /**
     * 获取是否包含偏置项
     * 
     * @return true表示包含偏置项，false表示不包含
     */
    public boolean isIncludeBias() {
        return this.includeBias;
    }
    
    /**
     * 获取正则化类型
     * 
     * @return 正则化类型
     */
    public RegularizationType getRegularizationType() {
        return this.regularizationType;
    }
    
    /**
     * 获取L1正则化系数
     * 
     * @return L1正则化系数
     */
    public double getLambda1() {
        return this.lambda1;
    }
    
    /**
     * 获取L2正则化系数
     * 
     * @return L2正则化系数
     */
    public double getLambda2() {
        return this.lambda2;
    }
    
    /**
     * 获取特征数量（不包括偏置项）
     * 
     * @return 特征数量
     */
    public int getFeatureCount() {
        return this.featureCount;
    }
    
    /**
     * 获取样本数量
     * 
     * @return 样本数量
     */
    public int getSampleCount() {
        return this.sampleCount;
    }
    
    /**
     * 设置优化器
     * 
     * @param optimizer 新的优化器
     */
    public void setOptimizer(IOptimizer optimizer) {
        this.optimizer = optimizer;
        applyLbfgsSettingsIfApplicable();
    }

    public int getMaxOptimizerIterations() {
        return maxOptimizerIterations;
    }

    public void setMaxOptimizerIterations(int maxOptimizerIterations) {
        this.maxOptimizerIterations = maxOptimizerIterations;
        applyLbfgsSettingsIfApplicable();
    }

    public double getOptimizerTolerance() {
        return optimizerTolerance;
    }

    public void setOptimizerTolerance(double optimizerTolerance) {
        this.optimizerTolerance = optimizerTolerance;
        applyLbfgsSettingsIfApplicable();
    }

    private void applyLbfgsSettingsIfApplicable() {
        if (optimizer instanceof RereLBFGS) {
            RereLBFGS lbfgs = (RereLBFGS) optimizer;
            lbfgs.setMaxIterations(maxOptimizerIterations);
            lbfgs.setTolerance(optimizerTolerance);
        }
    }
    
    /**
     * 设置正则化参数
     * 
     * @param regularizationType 正则化类型
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     * @throws IllegalArgumentException 如果正则化参数无效
     */
    public void setRegularization(RegularizationType regularizationType, double lambda1, double lambda2) {
        this.regularizationType = regularizationType;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        
        // 验证参数
        validateRegularizationParameters();
    }
    
    /**
     * 设置正则化系数（自动推断类型）
     * 
     * @param lambda1 L1正则化系数
     * @param lambda2 L2正则化系数
     */
    public void setRegularization(double lambda1, double lambda2) {
        updateRegularizationType(lambda1, lambda2);
    }
    
    /**
     * 设置L1正则化系数
     * 
     * @param lambda1 L1正则化系数
     */
    public void setLambda1(double lambda1) {
        this.lambda1 = lambda1;
        this.regularizationType = inferRegularizationType(this.lambda1, this.lambda2);
        validateRegularizationParameters();
    }
    
    /**
     * 设置L2正则化系数
     * 
     * @param lambda2 L2正则化系数
     */
    public void setLambda2(double lambda2) {
        this.lambda2 = lambda2;
        this.regularizationType = inferRegularizationType(this.lambda1, this.lambda2);
        validateRegularizationParameters();
    }
    
    /**
     * 获取正则化描述信息
     * 
     * @return 正则化描述字符串
     */
    public String getRegularizationDescription() {
        switch (regularizationType) {
            case L1:
                return String.format("L1正则化 (λ₁ = %.4f)", lambda1);
            case L2:
                return String.format("L2正则化 (λ₂ = %.4f)", lambda2);
            case ELASTIC_NET:
                return String.format("ElasticNet正则化 (λ₁ = %.4f, λ₂ = %.4f)", lambda1, lambda2);
            case NONE:
            default:
                return "无正则化";
        }
    }
    
    /**
     * 获取特征权重向量（不包括偏置项）
     * 
     * @return 特征权重向量，如果未训练则返回null
     */
    public IVector getFeatureWeights() {
        if (this.trainedWeights == null) {
            return null;
        }
        
        if (this.includeBias) {
            // 包含偏置项：权重向量的最后一个元素是偏置项
            // 使用向量切片获取除最后一个元素外的所有元素
            return this.trainedWeights.slice(0, this.trainedWeights.length() - 1);
        } else {
            // 不包含偏置项：整个权重向量都是特征权重
            return this.trainedWeights;
        }
    }
    
    /**
     * 获取偏置项
     * 
     * @return 偏置项值，如果不包含偏置项则返回0
     */
    public double getBias() {
        if (this.trainedWeights == null) {
            return 0.0;
        }
        
        if (this.includeBias) {
            // 包含偏置项：权重向量的最后一个元素是偏置项
            // 直接使用get方法替代手动数组访问
            return (double)this.trainedWeights.get(this.trainedWeights.length() - 1);
        } else {
            // 不包含偏置项
            return 0.0;
        }
    }
    
    /**
     * 获取完整的权重向量（包括偏置项，如果启用）
     * 
     * @return 完整的权重向量，如果未训练则返回null
     */
    public IVector getFullWeights() {
        return this.trainedWeights;
    }
    
    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        if (trainedWeights != null) p.put("trainedWeights", trainedWeights.toDoubleArray());
        p.put("includeBias", includeBias);
        p.put("regularizationType", regularizationType.name());
        p.put("lambda1", lambda1);
        p.put("lambda2", lambda2);
        p.put("featureCount", featureCount);
        p.put("sampleCount", sampleCount);
        p.put("maxOptimizerIterations", maxOptimizerIterations);
        p.put("optimizerTolerance", optimizerTolerance);
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        double[] wArr = (double[]) p.get("trainedWeights");
        if (wArr != null) this.trainedWeights = Linalg.vector(wArr);
        this.includeBias = (Boolean) p.get("includeBias");
        this.regularizationType = RegularizationType.valueOf((String) p.get("regularizationType"));
        this.lambda1 = ((Number) p.get("lambda1")).doubleValue();
        this.lambda2 = ((Number) p.get("lambda2")).doubleValue();
        this.featureCount = ((Number) p.get("featureCount")).intValue();
        this.sampleCount = ((Number) p.get("sampleCount")).intValue();
        if (p.containsKey("maxOptimizerIterations")) {
            this.maxOptimizerIterations = ((Number) p.get("maxOptimizerIterations")).intValue();
        }
        if (p.containsKey("optimizerTolerance")) {
            this.optimizerTolerance = ((Number) p.get("optimizerTolerance")).doubleValue();
        }
        applyLbfgsSettingsIfApplicable();
    }

    @Override
    public boolean isTrained() {
        return this.trainedWeights != null;
    }

    @Override
    public RegressionResult getResult() {
        return this.result;
    }
}
