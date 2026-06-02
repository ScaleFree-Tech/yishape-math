package com.yishape.lab.math.ml.dr;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.util.Tuple2;

import com.yishape.lab.util.YishapeLogger;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * 非负矩阵分解 (Non-negative Matrix Factorization)
 * <p>
 * NMF将一个非负矩阵X分解为两个非负矩阵W和H的乘积：X ≈ WH
 * - W: (nSamples x nComponents) 基矩阵
 * - H: (nComponents x nFeatures) 系数矩阵
 * </p>
 * <p>
 * 本实现参照sklearn的NMF，使用乘性更新算法（Multiplicative Update）。
 * </p>
 *
 * @author lteb2
 */
public class RereNMF implements ITransform<Double>, ISerializableModel {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereNMF.class);
    private static final double EPSILON = 1e-10;

    public enum InitMethod {
        RANDOM,
        NNDSVD
    }

    public enum BetaLoss {
        FROBENIUS,
        KULLBACK_LEIBLER,
        ITAKURA_SAITO
    }

    private int nComponents = -1;
    private InitMethod init = InitMethod.NNDSVD;
    private BetaLoss betaLoss = BetaLoss.FROBENIUS;
    private double tol = 1e-4;
    private int maxIter = 200;
    private int randomState = 42;
    private double alphaW = 0.0;
    private double alphaH = 0.0;
    private double l1Ratio = 0.0;
    private int verbose = 0;

    private double[][] W;
    private double[][] H;
    private double[][] X;
    private double reconstructionErr = -1;
    private int nIter = -1;
    private int nFeaturesIn;
    private boolean fitted = false;

    /** ITransform接口需要的字段 */
    private IMatrix<Double> feature;

    public RereNMF() {
    }

    public RereNMF(int nComponents) {
        this.nComponents = nComponents;
    }

    public RereNMF(int nComponents, InitMethod init) {
        this.nComponents = nComponents;
        this.init = init;
    }

    @Override
    public boolean ifTrained() {
        return fitted && feature != null;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null) {
            throw new IllegalArgumentException("特征数据不能为空");
        }

        int nSamples = feature.getRowNum();
        int nFeatures = feature.getColNum();
        this.feature = feature;
        this.nFeaturesIn = nFeatures;

        if (nComponents <= 0) {
            throw new IllegalStateException("必须先设置目标维度 / Target dimension must be set first (use setNComponents)");
        }

        this.X = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = (Double) feature.get(i, j);
            }
        }

        int effectiveComponents = nComponents;

        initializeW(nSamples, effectiveComponents);

        this.H = new double[effectiveComponents][nFeatures];

        double l1RegW = nFeatures * alphaW * l1Ratio;
        double l1RegH = nSamples * alphaH * l1Ratio;
        double l2RegW = nFeatures * alphaW * (1 - l1Ratio);
        double l2RegH = nSamples * alphaH * (1 - l1Ratio);

        double prevError = computeBetaDivergence(X, W, H);
        if (verbose > 0) {
            log.info("初始误差: {}", prevError);
        }

        for (int iter = 1; iter <= maxIter; iter++) {
            updateW(X, W, H, l1RegW, l2RegW);
            updateH(X, W, H, l1RegH, l2RegH);

            double error = computeBetaDivergence(X, W, H);

            if (verbose > 0 && iter % 10 == 0) {
                log.info("迭代 {}: 误差={}", iter, error);
            }

            if (Math.abs(prevError - error) / (prevError + EPSILON) < tol) {
                if (verbose > 0) {
                    log.info("在迭代 {} 收敛", iter);
                }
                nIter = iter;
                break;
            }

            prevError = error;
            nIter = iter;
        }

        if (nIter == maxIter) {
            log.warn("达到最大迭代次数 {}，未收敛", maxIter);
        }

        this.reconstructionErr = computeBetaDivergence(X, W, H);
        this.fitted = true;

        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!ifTrained()) {
            throw new IllegalStateException("模型尚未训练，请先调用fit / Model not trained, call fit first");
        }
        return IMatrix.of(W);
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    /**
     * 设置目标维度
     * @param nComponents 目标维度
     * @return 当前实例
     */
    public RereNMF setNComponents(int nComponents) {
        this.nComponents = nComponents;
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
     * 用NMF方法降维（便捷方法）
     * @param originalData 原数据矩阵
     * @param dim 目标维度
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim) {
        return setNComponents(dim).fit(originalData).transform(originalData);
    }

    private void initializeW(int nSamples, int nComponents) {
        Random rand = new Random(randomState);

        if (init == InitMethod.RANDOM) {
            double avg = 1.0;
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < X[0].length; j++) {
                    avg += X[i][j];
                }
            }
            avg /= (X.length * X[0].length);
            avg = Math.sqrt(avg / nComponents);

            this.W = new double[nSamples][nComponents];
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nComponents; j++) {
                    W[i][j] = Math.abs(rand.nextGaussian()) * avg;
                    if (W[i][j] < EPSILON) W[i][j] = EPSILON;
                }
            }
        } else {
            initializeNNDSVD(nSamples, nComponents);
        }
    }

    private void initializeNNDSVD(int nSamples, int nComponents) {
        double[][] XAbs = new double[nSamples][nFeaturesIn];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeaturesIn; j++) {
                XAbs[i][j] = Math.abs(X[i][j]);
            }
        }

        IMatrix<Double> XAbsMatrix = IMatrix.of(XAbs);
        Tuple2<IVector<Double>, IMatrix<Double>> svdResult = XAbsMatrix.eigen();
        double[] S = new double[svdResult._1.size()];
        for (int i = 0; i < S.length; i++) S[i] = (Double) svdResult._1.get(i);
        double[][] U = new double[svdResult._2.getRowNum()][svdResult._2.getColNum()];
        for (int i = 0; i < U.length; i++) {
            for (int j = 0; j < U[0].length; j++) {
                U[i][j] = (Double) svdResult._2.get(i, j);
            }
        }

        int effectiveComponents = Math.min(nComponents, S.length);

        this.W = new double[nSamples][effectiveComponents];
        this.H = new double[effectiveComponents][nFeaturesIn];

        for (int j = 0; j < effectiveComponents; j++) {
            double sqrtSj = Math.sqrt(Math.max(S[j], 0));
            for (int i = 0; i < nSamples; i++) {
                W[i][j] = sqrtSj * Math.abs(U[i][j]);
            }
            for (int i = 0; i < nFeaturesIn; i++) {
                H[j][i] = sqrtSj * Math.abs(U[j][i]);
            }
        }

        Random rand = new Random(randomState);
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < effectiveComponents; j++) {
                if (W[i][j] < EPSILON) {
                    W[i][j] = EPSILON + Math.abs(rand.nextGaussian()) * 0.01;
                }
            }
        }
        for (int i = 0; i < effectiveComponents; i++) {
            for (int j = 0; j < nFeaturesIn; j++) {
                if (H[i][j] < EPSILON) {
                    H[i][j] = EPSILON + Math.abs(rand.nextGaussian()) * 0.01;
                }
            }
        }
    }

    private void updateW(double[][] X, double[][] W, double[][] H,
                         double l1Reg, double l2Reg) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        int nComponents = W[0].length;

        double[][] WH = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0;
                for (int k = 0; k < nComponents; k++) {
                    sum += W[i][k] * H[k][j];
                }
                WH[i][j] = sum;
            }
        }

        double[][] numerator = new double[nSamples][nComponents];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nComponents; j++) {
                double sum = 0;
                for (int k = 0; k < nFeatures; k++) {
                    sum += X[i][k] * H[j][k] / Math.max(WH[i][k], EPSILON);
                }
                numerator[i][j] = sum;
            }
        }

        double[][] denominator = new double[nSamples][nComponents];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nComponents; j++) {
                double sum = 0;
                for (int k = 0; k < nFeatures; k++) {
                    sum += WH[i][k] * H[j][k];
                }
                denominator[i][j] = sum;
            }
        }

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nComponents; j++) {
                double denom = denominator[i][j] + l1Reg + l2Reg * W[i][j];
                if (denom > EPSILON) {
                    W[i][j] = W[i][j] * (numerator[i][j] / denom);
                }
                if (W[i][j] < EPSILON) W[i][j] = EPSILON;
            }
        }
    }

    private void updateH(double[][] X, double[][] W, double[][] H,
                         double l1Reg, double l2Reg) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        int nComponents = W[0].length;

        double[][] WH = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0;
                for (int k = 0; k < nComponents; k++) {
                    sum += W[i][k] * H[k][j];
                }
                WH[i][j] = sum;
            }
        }

        double[][] numerator = new double[nComponents][nFeatures];
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0;
                for (int k = 0; k < nSamples; k++) {
                    sum += W[k][i] * X[k][j] / Math.max(WH[k][j], EPSILON);
                }
                numerator[i][j] = sum;
            }
        }

        double[][] denominator = new double[nComponents][nFeatures];
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0;
                for (int k = 0; k < nSamples; k++) {
                    sum += W[k][i] * WH[k][j];
                }
                denominator[i][j] = sum;
            }
        }

        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double denom = denominator[i][j] + l1Reg + l2Reg * H[i][j];
                if (denom > EPSILON) {
                    H[i][j] = H[i][j] * (numerator[i][j] / denom);
                }
                if (H[i][j] < EPSILON) H[i][j] = EPSILON;
            }
        }
    }

    private double computeBetaDivergence(double[][] X, double[][] W, double[][] H) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        if (betaLoss == BetaLoss.FROBENIUS) {
            double sum = 0;
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    double wh = 0;
                    for (int k = 0; k < W[0].length; k++) {
                        wh += W[i][k] * H[k][j];
                    }
                    double diff = X[i][j] - wh;
                    sum += diff * diff;
                }
            }
            return Math.sqrt(sum);
        } else {
            double sum = 0;
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    double wh = 0;
                    for (int k = 0; k < W[0].length; k++) {
                        wh += W[i][k] * H[k][j];
                    }
                    if (X[i][j] > EPSILON && wh > EPSILON) {
                        sum += X[i][j] * Math.log(X[i][j] / wh) - X[i][j] + wh;
                    }
                }
            }
            return sum;
        }
    }

    public IMatrix getW() {
        return IMatrix.of(W);
    }

    public IMatrix getH() {
        return IMatrix.of(H);
    }

    public double getReconstructionError() {
        return reconstructionErr;
    }

    public int getNIter() {
        return nIter;
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("nComponents", nComponents);
        p.put("init", init.name());
        p.put("betaLoss", betaLoss.name());
        p.put("tol", tol);
        p.put("maxIter", maxIter);
        p.put("randomState", randomState);
        p.put("alphaW", alphaW);
        p.put("alphaH", alphaH);
        p.put("l1Ratio", l1Ratio);
        p.put("verbose", verbose);
        p.put("reconstructionErr", reconstructionErr);
        p.put("nIter", nIter);
        p.put("nFeaturesIn", nFeaturesIn);
        p.put("fitted", fitted);
        if (W != null) p.put("W", W.clone());
        if (H != null) p.put("H", H.clone());
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        this.nComponents = ((Number) p.get("nComponents")).intValue();
        this.init = InitMethod.valueOf((String) p.get("init"));
        this.betaLoss = BetaLoss.valueOf((String) p.get("betaLoss"));
        this.tol = ((Number) p.get("tol")).doubleValue();
        this.maxIter = ((Number) p.get("maxIter")).intValue();
        this.randomState = ((Number) p.get("randomState")).intValue();
        this.alphaW = ((Number) p.get("alphaW")).doubleValue();
        this.alphaH = ((Number) p.get("alphaH")).doubleValue();
        this.l1Ratio = ((Number) p.get("l1Ratio")).doubleValue();
        this.verbose = ((Number) p.get("verbose")).intValue();
        this.reconstructionErr = ((Number) p.get("reconstructionErr")).doubleValue();
        this.nIter = ((Number) p.get("nIter")).intValue();
        this.nFeaturesIn = ((Number) p.get("nFeaturesIn")).intValue();
        this.fitted = (Boolean) p.get("fitted");
        this.W = (double[][]) p.get("W");
        this.H = (double[][]) p.get("H");
    }

    public InitMethod getInit() { return init; }
    public void setInit(InitMethod init) { this.init = init; }
    public BetaLoss getBetaLoss() { return betaLoss; }
    public void setBetaLoss(BetaLoss betaLoss) { this.betaLoss = betaLoss; }
    public double getTol() { return tol; }
    public void setTol(double tol) { this.tol = tol; }
    public int getMaxIter() { return maxIter; }
    public void setMaxIter(int maxIter) { this.maxIter = maxIter; }
    public int getRandomState() { return randomState; }
    public void setRandomState(int randomState) { this.randomState = randomState; }
    public boolean isFitted() { return fitted; }
}
