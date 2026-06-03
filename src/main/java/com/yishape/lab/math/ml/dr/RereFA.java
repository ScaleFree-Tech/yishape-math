package com.yishape.lab.math.ml.dr;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.util.Tuple2;

import java.io.*;

/**
 * 因子分析 (Factor Analysis) 降维算法实现类
 * <p>
 * 实现基于最大似然估计的因子分析算法。因子分析是一种线性降维技术，
 * 它假设观测数据是由潜在的隐变量（因子）和噪声的线性组合生成的。
 * 与PCA不同，FA假设数据是由因子和噪声的线性组合生成的，并使用最大似然估计。
 * </p>
 *
 * <h3>算法步骤:</h3>
 * <ol>
 *   <li>数据中心化：X_centered = X - mean(X)</li>
 *   <li>初始化因子载荷矩阵和唯一性方差</li>
 *   <li>E步：计算隐因子后验分布的均值和协方差</li>
 *   <li>M步：更新参数以最大化期望对数似然</li>
 *   <li>迭代直到收敛</li>
 *   <li>返回降维后的因子得分</li>
 * </ol>
 *
 * @author lteb2
 */
public class RereFA implements ITransform<Double> {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereFA.class);

    private static final long serialVersionUID = 1L;

    /** 最大迭代次数 */
    private static final int DEFAULT_MAX_ITER = 100;

    /** 收敛阈值 */
    private static final double DEFAULT_TOLERANCE = 1e-4;

    /** 数值稳定性正则化 */
    private static final double REGULARIZATION = 1e-6;

    /** 因子数量 */
    private int nFactors;

    /** 最大迭代次数 */
    private int maxIter;

    /** 收敛阈值 */
    private double tolerance;

    /** 因子载荷矩阵 (p x k) */
    private IMatrix<Double> loadings;

    /** 唯一性方差向量 (p) */
    private IVector<Double> uniquenesses;

    /** 数据均值向量 (p) */
    private IVector<Double> mean;

    /** 是否已拟合 */
    private boolean fitted = false;

    /** 最后一次迭代的对数似然值 */
    private double lastLogLikelihood;

    /** 实际迭代次数 */
    private int iterations;

    /** ITransform接口需要的字段 */
    private IMatrix<Double> feature;
    private int nComponents = -1;

    /**
     * 使用默认参数创建因子分析器
     */
    public RereFA() {
        this(2);
    }

    /**
     * 创建因子分析器
     *
     * @param nFactors 因子数量
     */
    public RereFA(int nFactors) {
        this(nFactors, DEFAULT_MAX_ITER, DEFAULT_TOLERANCE);
    }

    /**
     * 创建因子分析器
     *
     * @param nFactors 因子数量
     * @param maxIter 最大迭代次数
     * @param tolerance 收敛阈值
     */
    public RereFA(int nFactors, int maxIter, double tolerance) {
        this.nFactors = nFactors;
        this.nComponents = nFactors;
        this.maxIter = maxIter;
        this.tolerance = tolerance;
    }

    @Override
    public boolean ifTrained() {
        return fitted && feature != null;
    }

    @Override
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null) {
            throw new IllegalArgumentException("特征数据不能为空 / Feature data cannot be null");
        }

        int nSamples = feature.getRowNum();
        int nFeatures = feature.getColNum();

        if (nSamples < 2) {
            throw new IllegalArgumentException("样本数量至少为2 / At least 2 samples required");
        }

        if (nComponents <= 0) {
            throw new IllegalArgumentException("目标维度必须大于0 / Target dimension must be positive");
        }

        if (nComponents >= nFeatures) {
            throw new IllegalArgumentException("目标维度必须小于原始数据的列数 / Target dimension must be less than original column count");
        }

        this.feature = feature;
        this.nFactors = nComponents;

        // 步骤1：数据中心化
        IVector<Double> colMeans = ((IMatrix)feature).colMeans();
        this.mean = colMeans;
        IMatrix<Double> centeredData = ((IMatrix)feature).broadcastSubRow(colMeans);

        // 步骤2：初始化参数
        initializeParameters(centeredData);

        // 步骤3：EM迭代
        emIteration(centeredData);

        fitted = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!ifTrained()) {
            throw new IllegalStateException("模型尚未训练，请先调用fit / Model not trained, call fit first");
        }

        if (feature.getColNum() != this.feature.getColNum()) {
            throw new IllegalArgumentException("特征维度不匹配 / Feature dimension mismatch");
        }

        // 步骤1：使用训练集均值进行数据中心化
        IMatrix<Double> centeredData = ((IMatrix)feature).broadcastSubRow(this.mean);

        // 步骤4：计算因子得分（隐变量后验均值）
        IMatrix<Double> factorScores = computeFactorScores(centeredData);

        return factorScores;
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    /**
     * 设置目标维度（因子数量）
     * @param nComponents 目标维度
     * @return 当前实例
     */
    public RereFA setNComponents(int nComponents) {
        this.nComponents = nComponents;
        this.nFactors = nComponents;
        return this;
    }

    /**
     * 获取目标维度
     * @return 目标维度
     */
    public int getNComponents() {
        return nComponents;
    }

    /**
     * 执行因子分析降维（便捷方法）
     * <p>
     * 通过最大似然估计的因子分析对输入数据进行降维处理。
     * </p>
     *
     * @param originalData 原数据矩阵，每行表示一个样本，每列表示一个特征
     * @param dim 目标维度（因子数量），必须小于原始数据的列数
     * @return 降维后的矩阵，每列为一个因子的得分
     */
    public IMatrix<Double> dimensionReduction(IMatrix originalData, int dim) {
        return setNComponents(dim).fit(originalData).transform(originalData);
    }

    /**
     * 初始化因子载荷矩阵和唯一性方差
     */
    private void initializeParameters(IMatrix<Double> data) {
        int nFeatures = data.getColNum();

        // 使用PCA初始化：取前nFactors个主成分方向作为初始因子载荷
        // data 已经是中心化后的数据
        IMatrix<Double> cov = (IMatrix<Double>) data.covarianceFromCentered();
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = cov.eigen();
        IVector<Double> eigenValues = (IVector<Double>) eigenResult._1;
        IMatrix<Double> eigenVectors = (IMatrix<Double>) eigenResult._2;

        // 取前nFactors个特征向量，缩放后作为初始载荷
        IMatrix<Double> initLoadings = (IMatrix<Double>) eigenVectors.slice(0, nFeatures, 0, nFactors);
        double[] sqrtEigenArr = eigenValues.toDoubleArray();
        for (int j = 0; j < nFactors; j++) {
            double sqrtEigen = Math.max(Math.sqrt(sqrtEigenArr[j]), REGULARIZATION);
            for (int i = 0; i < nFeatures; i++) {
                initLoadings.set(i, j, initLoadings.get(i, j) * sqrtEigen);
            }
        }
        this.loadings = initLoadings;

        // 初始化唯一性方差为1减去共同度（近似）
        IVector<Double> communalities = this.loadings.mmul(this.loadings.transpose()).diag();
        double[] uniquenessArr = new double[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            uniquenessArr[i] = Math.max(1.0 - communalities.get(i), 0.1);
        }
        this.uniquenesses = IVector.of(uniquenessArr);
    }

    /**
     * EM迭代
     */
    private void emIteration(IMatrix<Double> data) {
        int nSamples = data.getRowNum();
        int nFeatures = data.getColNum();

        for (int iter = 0; iter < maxIter; iter++) {
            // E步：计算隐因子后验
            // 后验协方差：Sigma_z = (I + L^T Psi^{-1} L)^{-1}
            // 后验均值：Mu_z = Sigma_z L^T Psi^{-1} X

            // 构建 Psi (唯一性方差对角矩阵)
            IMatrix<Double> psiInv = IMatrix.diag(
                java.util.stream.IntStream.range(0, nFeatures)
                    .mapToDouble(i -> 1.0 / Math.max(uniquenesses.get(i), REGULARIZATION))
                    .toArray()
            );

            // L^T Psi^{-1} L
            IMatrix<Double> ltPsiInv = loadings.transpose().mmul(psiInv);
            IMatrix<Double> ltPsiInvL = ltPsiInv.mmul(loadings);

            // Sigma_z = (I + L^T Psi^{-1} L)^{-1}
            IMatrix<Double> identity = IMatrix.eye(nFactors);
            IMatrix<Double> sigmaZ = (IMatrix<Double>) identity.add(ltPsiInvL).inv();

            // Mu_z = Sigma_z L^T Psi^{-1} X^T (对每个样本)
            IMatrix<Double> psiInvX = psiInv.mmul(data.transpose());
            IMatrix<Double> muZ = sigmaZ.mmul(loadings.transpose()).mmul(psiInvX).transpose();

            // M步：更新参数
            // 更新载荷：L = X^T Z (Z^T Z)^{-1}
            IMatrix<Double> ztz = muZ.transpose().mmul(muZ);
            // 添加正则化确保可逆
            ztz = (IMatrix<Double>) ztz.add(Linalg.eye(nFactors).multiplyByScalar(REGULARIZATION));
            IMatrix<Double> ztzInv = (IMatrix<Double>) ztz.inv();
            IMatrix<Double> xtz = data.transpose().mmul(muZ);
            IMatrix<Double> newLoadings = xtz.mmul(ztzInv);

            // 更新唯一性方差
            // Psi_ii = Var(X_i - Z_i L_i) = (1/n) * sum((x_i - z_i L_i)^2)
            IMatrix<Double> residuals = data.sub(muZ.mmul(loadings.transpose()));
            double[] varArr = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                double sumSq = 0;
                for (int i = 0; i < nSamples; i++) {
                    double r = residuals.get(i, j);
                    sumSq += r * r;
                }
                varArr[j] = sumSq / nSamples;
            }

            // 确保唯一性不为负且不会太小
            for (int i = 0; i < nFeatures; i++) {
                varArr[i] = Math.max(varArr[i], REGULARIZATION);
            }

            this.loadings = newLoadings;
            this.uniquenesses = IVector.of(varArr);

            // 检查收敛
            double logLike = computeLogLikelihood(data, sigmaZ, muZ);
            if (iter > 0 && Math.abs(logLike - lastLogLikelihood) < tolerance) {
                lastLogLikelihood = logLike;
                iterations = iter + 1;
                return;
            }
            lastLogLikelihood = logLike;
        }
        iterations = maxIter;
    }

    /**
     * 计算对数似然
     */
    private double computeLogLikelihood(IMatrix<Double> data, IMatrix<Double> sigmaZ, IMatrix<Double> muZ) {
        int n = data.getRowNum();
        int p = data.getColNum();

        // 简化的对数似然计算
        // LL = -0.5 * n * (p * log(2*pi) + log(|Sigma|) + tr(Sigma^{-1} * S))
        // 其中 Sigma = L L^T + Psi, S 是样本协方差

        IMatrix<Double> llLt = loadings.mmul(loadings.transpose());
        IMatrix<Double> sigma = (IMatrix<Double>) llLt.add(IMatrix.diag(uniquenesses.toDoubleArray()));

        // 计算行列式（简化）
        double logDet = 0;
        for (int i = 0; i < p; i++) {
            logDet += Math.log(Math.max(sigma.get(i, i), REGULARIZATION));
        }

        // 简化：使用迹的近似
        double traceTerm = 0;
        for (int i = 0; i < p; i++) {
            traceTerm += 1.0;
        }

        double logLike = -0.5 * n * (p * Math.log(2 * Math.PI) + logDet + traceTerm);
        return logLike;
    }

    /**
     * 计算因子得分（隐变量后验均值）
     */
    private IMatrix<Double> computeFactorScores(IMatrix<Double> data) {
        int nFeatures = data.getColNum();

        // Psi^{-1}
        IMatrix<Double> psiInv = IMatrix.diag(
            java.util.stream.IntStream.range(0, nFeatures)
                .mapToDouble(i -> 1.0 / Math.max(uniquenesses.get(i), REGULARIZATION))
                .toArray()
        );

        // Sigma_z = (I + L^T Psi^{-1} L)^{-1}
        IMatrix<Double> ltPsiInv = loadings.transpose().mmul(psiInv);
        IMatrix<Double> identity = IMatrix.eye(nFactors);
        IMatrix<Double> sigmaZ = (IMatrix<Double>) identity.add(ltPsiInv.mmul(loadings)).inv();

        // Mu_z = Sigma_z L^T Psi^{-1} X^T
        IMatrix<Double> psiInvX = psiInv.mmul(data.transpose());
        IMatrix<Double> factorScores = sigmaZ.mmul(loadings.transpose()).mmul(psiInvX).transpose();

        return factorScores;
    }

    /**
     * 获取因子载荷矩阵
     */
    public IMatrix<Double> getLoadings() {
        return loadings;
    }

    /**
     * 获取唯一性方差
     */
    public IVector<Double> getUniquenesses() {
        return uniquenesses;
    }

    /**
     * 获取数据均值
     */
    public IVector<Double> getMean() {
        return mean;
    }

    /**
     * 是否已拟合
     */
    public boolean isFitted() {
        return fitted;
    }

    /**
     * 获取实际迭代次数
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * 将模型保存在本地
     *
     * @param path 保存路径
     */
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            log.error("保存模型失败 / Failed to save model", e);
        }
    }
}
