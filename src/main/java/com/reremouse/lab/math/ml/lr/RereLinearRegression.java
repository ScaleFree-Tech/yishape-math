package com.reremouse.lab.math.ml.lr;

import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

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
 * 3. 目标函数：均方误差损失 + 正则化项（L1、L2或ElasticNet）
 * 4. 自动处理特征矩阵的增广（添加偏置列）
 * </p>
 * 
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建线性回归模型
 RereLinearRegression lr = new RereLinearRegression();
 
 // 训练模型
 RegressionResult result = lr.fit(featureMatrix, labelVector);
 
 // 预测新样本
 double prediction = lr.predict(newFeatureVector);
 
 // 获取模型权重
 IVector weights = result.getWeights();
 double loss = result.getLoss();
 }
 * </pre>
 * 
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class RereLinearRegression implements IRegression, IGradientFunction, IObjectiveFunction {
    
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
     * 训练后的权重向量（包含偏置项）
     */
    private IVector trainedWeights = null;
    
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

    @Override
    /**
     * 训练线性回归模型
     * <p>
     * 训练过程：
     * 1. 数据预处理：增广特征矩阵（添加偏置列）
     * 2. 初始化权重：零向量或随机初始化
     * 3. 使用优化器求解最优权重
     * 4. 计算最终损失值
     * 5. 返回训练结果
     * </p>
     * 
     * @param feature 特征矩阵，每行是一个样本，每列是一个特征
     * @param labels 标签向量，对应每个样本的真实值
     * @return 训练结果，包含最优权重和损失值
     * @throws IllegalArgumentException 如果输入参数无效
     */
    public RegressionResult fit(IMatrix feature, IVector labels) {
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
        
        // 使用优化器求解最优权重
        Tuple2<Double, IVector> optimizationResult = optimizer.optimize(
            initialWeights, this, this);
        
        // 保存训练结果
        this.trainedWeights = optimizationResult._2;
        double finalLoss = optimizationResult._1;
        
        // 创建并返回训练结果
        RegressionResult result = new RegressionResult();
        
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
        
        return result;
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
        double prediction = (double)this.trainedWeights.innerProduct(augmentedX);
        
        return prediction;
    }

    @Override
    /**
     * 计算目标函数的梯度
     * <p>
     * 对于线性回归，目标函数为：
     * J(w) = (1/2n) * ||X*w - y||^2 + R(w)
     * 
     * 其中R(w)是正则化项：
     * - L1正则化：R(w) = λ₁ * ||w||₁
     * - L2正则化：R(w) = (λ₂/2) * ||w||²
     * - ElasticNet：R(w) = λ₁ * ||w||₁ + (λ₂/2) * ||w||²
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
        double mse = (double)residuals.innerProduct(residuals) / (2.0 * this.sampleCount);
        
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
                return lambda1 * computeL1Norm(w);
            case L2:
                return lambda2 * (double)w.innerProduct(w) / 2.0;
            case ELASTIC_NET:
                return lambda1 * computeL1Norm(w) + lambda2 * (double)w.innerProduct(w) / 2.0;
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
                return computeL1Gradient(w);
            case L2:
                return (IVector)w.multiplyScalar(lambda2);
            case ELASTIC_NET:
                return computeElasticNetGradient(w);
            case NONE:
            default:
                return null;
        }
    }
    
    /**
     * 计算L1范数：||w||₁ = Σ|wᵢ|
     * 
     * @param w 权重向量
     * @return L1范数值
     */
    private double computeL1Norm(IVector w) {
        // 使用向量内置的L1范数计算方法
        return (double)w.norm1();
    }
    
    /**
     * 计算L1正则化的梯度
     * <p>
     * L1正则化的梯度是符号函数：∇||w||₁ = sign(w)
     * 其中sign(wᵢ) = 1 if wᵢ > 0, -1 if wᵢ < 0, 0 if wᵢ = 0
     * </p>
     * 
     * @param w 权重向量
     * @return L1正则化的梯度向量
     */
    private IVector computeL1Gradient(IVector w) {
        // 使用向量的sign方法计算L1正则化梯度，替代手动循环
        IVector signVector = w.sign();
        return (IVector)signVector.multiplyScalar(lambda1);
    }
    
    /**
     * 计算ElasticNet正则化的梯度
     * <p>
     * ElasticNet = L1 + L2，所以梯度是两者的组合：
     * ∇R(w) = λ₁ * sign(w) + λ₂ * w
     * </p>
     * 
     * @param w 权重向量
     * @return ElasticNet正则化的梯度向量
     */
    private IVector computeElasticNetGradient(IVector w) {
        // L1部分
        IVector l1Gradient = computeL1Gradient(w);
        
        // L2部分
        IVector l2Gradient = (IVector)w.multiplyScalar(lambda2);
        
        // 组合梯度
        return (IVector)l1Gradient.add(l2Gradient);
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
        return gradient.multiplyScalar(1.0 / this.sampleCount);
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
}
