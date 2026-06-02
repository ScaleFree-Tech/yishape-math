package com.yishape.lab.math.ml.dml.gmml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.MetricTransforms;
import com.yishape.lab.util.Tuple2;

import java.util.HashSet;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Geometric Mean Metric Learning (GMML)：基于黎曼几何的半正定矩阵流形上的测地线计算，
 * 在相似矩阵与异类矩阵之间插值得到最优马氏度量。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Zadeh, P., et al. (2016). Geometric mean metric learning.
 *       In <em>ICML</em>, pp. 2464–2471.</li>
 * </ul>
 */
public final class GmmlDml implements ISupervisedDml {

    private double geodesicStep = 0.5;
    private double reg = 1e-6;
    private int constraintFactor = 40;
    private Object prior = null;
    private double autoThresh = 1e-9;
    private Random random;

    public double getGeodesicStep() {
        return geodesicStep;
    }

    public GmmlDml setGeodesicStep(double geodesicStep) {
        if (geodesicStep < 0 || geodesicStep > 1) {
            throw new IllegalArgumentException("geodesicStep 须在 [0,1] 范围内");
        }
        this.geodesicStep = geodesicStep;
        return this;
    }

    public double getReg() {
        return reg;
    }

    public GmmlDml setReg(double reg) {
        this.reg = reg;
        return this;
    }

    public int getConstraintFactor() {
        return constraintFactor;
    }

    public GmmlDml setConstraintFactor(int constraintFactor) {
        this.constraintFactor = constraintFactor;
        return this;
    }

    public Object getPrior() {
        return prior;
    }

    public GmmlDml setPrior(Object prior) {
        this.prior = prior;
        return this;
    }

    public GmmlDml setPriorIdentity() {
        this.prior = "identity";
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public GmmlDml setRandom(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        return fit(features, DmlArrays.stringLabels(labels));
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, GmmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, GmmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        Set<Integer> uniqueClasses = new HashSet<>();
        for (int yi : y) {
            uniqueClasses.add(yi);
        }
        int numClasses = uniqueClasses.size();

        int numConst = constraintFactor * numClasses * (numClasses - 1);

        double[][] S = generateConstraints(x, y, numConst, rnd, true);
        double[][] D = generateConstraints(x, y, numConst, rnd, false);

        IMatrix<Double> sMat = IMatrix.of(S);
        IMatrix<Double> dMat = IMatrix.of(D);

        if (prior != null && "identity".equals(prior)) {
            IMatrix<Double> regMat = Linalg.diag(new double[d]).multiplyByScalar(reg);
            sMat = sMat.add(regMat);
            dMat = dMat.add(regMat);
        }

        IMatrix<Double> sInv = sMat.inv();
        IMatrix<Double> result = interpolateMatrices(sInv, dMat, geodesicStep);

        IMatrix<Double> metric = MetricTransforms.symmetrize(result);
        return DmlMetric.fullWhitening(metric);
    }

    private double[][] generateConstraints(double[][] x, int[] y, int numConst, Random rnd, boolean sameClass) {
        int n = x.length;
        int d = x[0].length;

        double[][] aggregated = new double[d][d];
        int generated = 0;
        int attempts = 0;
        int maxAttempts = numConst * 8;

        while (generated < numConst && attempts < maxAttempts) {
            attempts++;
            int idx1 = rnd.nextInt(n);
            int idx2 = rnd.nextInt(n);

            if (sameClass ? (y[idx1] != y[idx2]) : (y[idx1] == y[idx2])) {
                continue;
            }

            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    double diff = x[idx1][a] - x[idx2][a];
                    aggregated[a][b] += diff * (x[idx1][b] - x[idx2][b]);
                }
            }
            generated++;
        }

        if (generated > 0) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    aggregated[a][b] /= generated;
                }
            }
        }

        return aggregated;
    }

    /**
     * 基于 Cholesky-Schur 算法的完整测地线插值。
     * 等价于 pyDML: GMML._compute_geodesic_point(A, B, t)
     *
     * 步骤：
     * 1. 保证 cond(A) >= cond(B)，否则交换并调整 t
     * 2. R_A = chol(A).T, R_B = chol(B).T
     * 3. Z = R_B @ R_A^{-1}
     * 4. Z^T @ Z 的特征分解：D, U（按特征值升序）
     * 5. T = diag(D)^{t/2} @ U^T @ R_A
     * 6. G = T^T @ T
     */
    private IMatrix<Double> interpolateMatrices(IMatrix<Double> A, IMatrix<Double> B, double t) {
        int d = A.getRowNum();

        // 步骤 1：确保 cond(A) >= cond(B)，否则交换
        double condA = estimateCondition(A);
        double condB = estimateCondition(B);
        double tAdjusted = t;
        IMatrix<Double> Awork = A;
        IMatrix<Double> Bwork = B;

        if (condA < condB) {
            // 交换 A 和 B
            IMatrix<Double> tmp = Awork;
            Awork = Bwork;
            Bwork = tmp;
            tAdjusted = 1.0 - t;
        }

        // 步骤 2：Cholesky 分解
        IMatrix<Double> R_A = Awork.cholesky(); // 下三角
        IMatrix<Double> R_B = Bwork.cholesky();

        // 步骤 3：Z = R_B @ R_A^{-1} = R_B @ inv(R_A)
        // 对于下三角矩阵直接用通用 inv()，数值稳定性良好
        IMatrix<Double> R_A_inv = R_A.inv();
        IMatrix<Double> Z = R_B.mmul(R_A_inv);

        // 步骤 4：Z^T @ Z 的特征分解
        IMatrix<Double> ZtZ = Z.transpose().mmul(Z);
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = ZtZ.eigen();
        IVector<Double> evals = eigenResult._1;
        IMatrix<Double> evecs = eigenResult._2; // 列是特征向量

        // 按特征值升序排列（eigen 可能已排序，需确认）
        int rank = evals.size();
        double[] sortedEvals = new double[rank];
        double[][] sortedVects = new double[rank][rank];
        Integer[] indices = new Integer[rank];
        for (int i = 0; i < rank; i++) indices[i] = i;
        java.util.Arrays.sort(indices, (a, b) -> Double.compare((Double) evals.get(a), (Double) evals.get(b)));
        for (int i = 0; i < rank; i++) {
            sortedEvals[i] = (Double) evals.get(indices[i]);
            for (int j = 0; j < rank; j++) {
                sortedVects[i][j] = (Double) evecs.get(j, indices[i]);
            }
        }

        // 步骤 5：T = diag(D^{t/2}) @ U^T @ R_A
        // D^{t/2} = diag(sortedEvals[i]^(tAdjusted/2))
        final double tFinal = tAdjusted;
        IMatrix<Double> D_half = IMatrix.diag(java.util.stream.DoubleStream.of(sortedEvals)
                .map(v -> Math.pow(v, tFinal / 2.0)).toArray());

        // U^T (sortedVects 的转置，列是特征向量，所以 U^T 的行是特征向量)
        IMatrix<Double> U_T = IMatrix.of(sortedVects).transpose();

        // T = D_half @ U_T @ R_A
        IMatrix<Double> T = D_half.mmul(U_T).mmul(R_A);

        // 步骤 6：G = T^T @ T
        return T.transpose().mmul(T);
    }

    /**
     * 估算矩阵条件数（最大奇异值/最小奇异值的比值）。
     * 使用特征值估计（对 SPD 矩阵即条件数）。
     */
    private double estimateCondition(IMatrix<Double> M) {
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = M.eigen();
        IVector<Double> evals = eigenResult._1;
        double maxEval = Double.NEGATIVE_INFINITY;
        double minEval = Double.POSITIVE_INFINITY;
        for (int i = 0; i < evals.size(); i++) {
            double v = Math.abs((Double) evals.get(i));
            if (v > maxEval) maxEval = v;
            if (v > 0 && v < minEval) minEval = v;
        }
        if (minEval <= 0) minEval = 1e-15;
        return maxEval / minEval;
    }
}
