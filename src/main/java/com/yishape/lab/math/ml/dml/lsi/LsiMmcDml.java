package com.yishape.lab.math.ml.dml.lsi;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.ISupervisedDml;
import com.yishape.lab.util.Tuple2;

import java.util.Objects;

/**
 * Mahalanobis Metric for Clustering (MMC)：监督型马氏度量学习。
 *
 * <p><strong>对应 pyDML</strong>：此类对应 pyDML 的 LSI 算法（尽管命名不同）。</p>
 *
 * <p>MMC 通过最小化同类样本对之间的平方距离，同时确保异类样本对之间的距离足够大
 * （使用对数势垒约束），来学习一个正半定（PSD）马氏度量矩阵 {@code A}。
 * 这是一个凸优化问题，理论上保证全局最优解。</p>
 *
 * <p>本类实现 {@link ISupervisedDml}，需要标签来构造相似/不相似约束。</p>
 *
 * <h2>算法细节</h2>
 * <ul>
 *   <li><b>目标</b>：最小化 {@code f_S(A) = Σ_{i,j∈S} (x_i - x_j)'A(x_i - x_j)}（同类距离）</li>
 *   <li><b>约束</b>：{@code f_D(A) ≥ 1}（异类距离，使用对数势垒）</li>
 *   <li><b>约束</b>：{@code A ⪰ 0}（PSD）</li>
 * </ul>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Xing, E. P., Ng, A. Y., Jordan, M. I., &amp; Russell, S. (2002).
 *       Distance metric learning with application to clustering with side-information.
 *       In <em>NeurIPS 15</em>.</li>
 *   <li>智勇, 陈浩. "基于马氏度量聚类的距离度量学习".</li>
 * </ul>
 */
public final class LsiMmcDml implements ISupervisedDml {

    private int maxIter = 100;
    private double tolerance = 1e-3;
    private boolean diagonal = false;
    private double diagonalC = 1.0;

    public LsiMmcDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public LsiMmcDml setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public double getTolerance() {
        return tolerance;
    }

    public LsiMmcDml setDiagonal(boolean diagonal) {
        this.diagonal = diagonal;
        return this;
    }

    public boolean isDiagonal() {
        return diagonal;
    }

    public LsiMmcDml setDiagonalC(double diagonalC) {
        this.diagonalC = diagonalC;
        return this;
    }

    public double getDiagonalC() {
        return diagonalC;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, LsiMmcDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, LsiMmcDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        if (diagonal) {
            return fitDiagonal(x, y, n, d);
        } else {
            return fitFull(x, y, n, d);
        }
    }

    private DmlMetric fitFull(double[][] x, int[] y, int n, int d) {
        // 构造相似对和不相似对
        double[][] posPairs = null;
        double[][] negPairs = null;
        int nPos = 0;
        int nNeg = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (y[i] == y[j]) {
                    nPos++;
                } else {
                    nNeg++;
                }
            }
        }

        posPairs = new double[nPos][d];
        negPairs = new double[nNeg][d];
        int pi = 0, ni = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double[] diff = new double[d];
                for (int k = 0; k < d; k++) {
                    diff[k] = x[i][k] - x[j][k];
                }
                if (y[i] == y[j]) {
                    posPairs[pi++] = diff;
                } else {
                    negPairs[ni++] = diff;
                }
            }
        }

        // 初始化 A 为 d×d 单位矩阵
        double[][] A = new double[d][d];
        for (int i = 0; i < d; i++) {
            A[i][i] = 1.0;
        }

        // 迭代优化
        double eps = 0.01;
        double alpha = 0.1;
        double[][] A_old = new double[d][d];
        double[][] grad1 = computeFs1Grad(posPairs, d);
        double[][] grad2 = computeFd1Grad(negPairs, A, d);
        double[][] M = gradProjection(grad1, grad2);

        for (int cycle = 0; cycle < maxIter; cycle++) {
            // PSD 投影迭代
            boolean satisfy = false;
            for (int it = 0; it < 10000; it++) {
                // 计算 w 和 t
                double[] w = computeW(posPairs, d);
                double t = 0.0;
                for (int idx = 0; idx < w.length; idx++) {
                    for (int j = 0; j < d; j++) {
                        for (int k = 0; k < d; k++) {
                            t += w[idx] * A[j][k];
                        }
                    }
                    t /= 100.0;
                }

                // 计算 w_norm, w1, t1
                double wNorm = 0.0;
                for (double v : w) {
                    wNorm += v * v;
                }
                wNorm = Math.sqrt(wNorm);
                if (wNorm < 1e-10) wNorm = 1.0;

                double[] w1 = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    w1[i] = w[i] / wNorm;
                }
                double t1 = t / wNorm;

                // 第一个约束投影
                double[] x0 = flatten(A, d, d);
                double wDotX0 = 0.0;
                for (int i = 0; i < w.length; i++) {
                    wDotX0 += w1[i] * x0[i];
                }
                double[] xProj;
                if (wDotX0 <= t1) {
                    xProj = x0;
                } else {
                    xProj = new double[x0.length];
                    for (int i = 0; i < x0.length; i++) {
                        xProj[i] = x0[i] + (t1 - wDotX0) * w1[i];
                    }
                }

                // 更新 A
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A[i][j] = xProj[i * d + j];
                    }
                }

                // 第二个约束：PSD 投影
                double[][] AT = new double[d][d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        AT[i][j] = (A[i][j] + A[j][i]) / 2.0;
                    }
                }
                IMatrix<Double> mat = IMatrix.of(AT);
                Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = mat.eigen();
                IVector<Double> evals = eigenResult._1;
                IMatrix<Double> evecs = eigenResult._2;

                double[][] A_psd = new double[d][d];
                for (int i = 0; i < d; i++) {
                    double eigVal = Math.max(0.0, evals.get(i));
                    for (int j = 0; j < d; j++) {
                        A_psd[i][j] = eigVal * evecs.get(i, j);
                    }
                }
                // A_psd = V * diag(max(0, λ)) * V^T
                double[][] V = new double[d][d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        V[i][j] = evecs.get(i, j);
                    }
                }
                double[][] diagMat = new double[d][d];
                for (int i = 0; i < d; i++) {
                    diagMat[i][i] = Math.max(0.0, evals.get(i));
                }
                // A_psd = V * diagMat * V^T
                double[][] tmp = new double[d][d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        for (int k = 0; k < d; k++) {
                            tmp[i][j] += V[i][k] * diagMat[k][j];
                        }
                    }
                }
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A_psd[i][j] = 0.0;
                        for (int k = 0; k < d; k++) {
                            A_psd[i][j] += tmp[i][k] * V[j][k];
                        }
                    }
                }
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A[i][j] = A_psd[i][j];
                    }
                }

                // 检查第二个约束是否满足
                double fDC2 = 0.0;
                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < d; j++) {
                        for (int k = 0; k < d; k++) {
                            fDC2 += w[i] * A[j][k];
                        }
                    }
                }
                double error2 = (fDC2 - t) / (Math.abs(t) + 1e-10);
                if (error2 < eps) {
                    satisfy = true;
                    break;
                }
            }

            // 梯度上升步骤
            copyMatrix(A, A_old);
            double objOld = computeFdObj(negPairs, A_old);
            double objNew = computeFdObj(negPairs, A);

            if (satisfy && (objNew > objOld || cycle == 0)) {
                // 投影成功且目标改善
                alpha *= 1.05;
                grad2 = computeFs1Grad(posPairs, A, d);
                grad1 = computeFd1Grad(negPairs, A, d);
                M = gradProjection(grad1, grad2);
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A[i][j] += alpha * M[i][j];
                    }
                }
            } else {
                // 投影失败或目标退化
                alpha /= 2.0;
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A[i][j] = A_old[i][j] + alpha * M[i][j];
                    }
                }
            }

            // 检查收敛
            double delta = 0.0;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    double diff = alpha * M[i][j];
                    delta += diff * diff;
                }
            }
            delta = Math.sqrt(delta);
            double A_oldNorm = 0.0;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    A_oldNorm += A_old[i][j] * A_old[i][j];
                }
            }
            A_oldNorm = Math.sqrt(A_oldNorm);
            delta /= (A_oldNorm + 1e-10);

            if (delta < tolerance) {
                break;
            }
        }

        // 最终对称化
        for (int i = 0; i < d; i++) {
            for (int j = i + 1; j < d; j++) {
                A[i][j] = (A[i][j] + A[j][i]) / 2.0;
                A[j][i] = A[i][j];
            }
        }

        return DmlMetric.fullWhitening(Linalg.matrix(A));
    }

    private DmlMetric fitDiagonal(double[][] x, int[] y, int n, int d) {
        // 对角版本：学习对角矩阵 A = diag(w)
        // 使用 Newton-Raphson 更新
        double[] w = new double[d];
        for (int i = 0; i < d; i++) {
            w[i] = 1.0;
        }

        int it = 0;
        double error = 1.0;
        double eps = 1e-6;
        int reduction = 2;

        // 计算 s_sum = Σ_{i,j∈S} (x_i - x_j)^2
        double[] sSum = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (y[i] == y[j]) {
                    for (int k = 0; k < d; k++) {
                        double diff = x[i][k] - x[j][k];
                        sSum[k] += diff * diff;
                    }
                }
            }
        }

        while (error > tolerance && it < maxIter) {
            // 计算 fD, fD_1st, fD_2nd
            double[] fD_1st = new double[d];
            double[][] fD_2nd = new double[d][d];
            double sumDist = 0.0;

            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (y[i] != y[j]) {
                        double dist = 0.0;
                        double[] diffSq = new double[d];
                        for (int k = 0; k < d; k++) {
                            double diff = x[i][k] - x[j][k];
                            diffSq[k] = diff * diff;
                            dist += diffSq[k] * w[k];
                        }
                        dist = Math.sqrt(Math.max(dist, 1e-10));
                        sumDist += dist;
                        for (int k = 0; k < d; k++) {
                            fD_1st[k] += diffSq[k] / (2.0 * dist);
                        }
                    }
                }
            }

            double fD0 = Math.log(sumDist + 1e-6);
            double objInitial = 0.0;
            for (int k = 0; k < d; k++) {
                objInitial += sSum[k] * w[k];
            }
            objInitial += diagonalC * fD0;

            // 梯度 = sSum - diagonalC * fD_1st
            double[] gradient = new double[d];
            for (int k = 0; k < d; k++) {
                gradient[k] = sSum[k] - diagonalC * fD_1st[k];
            }

            // Hessian = -diagonalC * fD_2nd + eps * I
            double[][] hessian = new double[d][d];
            for (int k = 0; k < d; k++) {
                hessian[k][k] = eps;
            }

            // 求解 Newton 步
            double[] step = new double[d];
            for (int k = 0; k < d; k++) {
                step[k] = gradient[k] / (hessian[k][k] + 1e-10);
            }

            // 线搜索
            double lambda = 1.0;
            double[] wTmp = new double[d];
            double objPrev = Double.POSITIVE_INFINITY;

            for (int innerIt = 0; innerIt < 100; innerIt++) {
                for (int k = 0; k < d; k++) {
                    wTmp[k] = Math.max(0.0, w[k] - lambda * step[k]);
                }
                double obj = 0.0;
                for (int k = 0; k < d; k++) {
                    obj += sSum[k] * wTmp[k];
                }

                double sumDistTmp = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        if (y[i] != y[j]) {
                            double dist = 0.0;
                            for (int k = 0; k < d; k++) {
                                double diff = x[i][k] - x[j][k];
                                dist += diff * diff * wTmp[k];
                            }
                            sumDistTmp += Math.sqrt(Math.max(dist, 1e-10));
                        }
                    }
                }
                obj += diagonalC * Math.log(sumDistTmp + 1e-6);

                if (obj < objPrev) {
                    objPrev = obj;
                    lambda /= reduction;
                } else {
                    break;
                }
            }

            for (int k = 0; k < d; k++) {
                w[k] = Math.max(0.0, w[k] - lambda * step[k]);
            }

            error = Math.abs((objPrev - objInitial) / (objPrev + 1e-10));
            it++;
        }

        return DmlMetric.diagonal(Linalg.vector(w), 0, 0);
    }

    // ---- 辅助方法 ----

    /**
     * 计算相似约束梯度：∇f_S = Σ d_{ij}⊗d_{ij} for (i,j)∈S
     */
    private static double[][] computeFs1Grad(double[][] posPairs, int d) {
        double[][] grad = new double[d][d];
        for (double[] pair : posPairs) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    grad[i][j] += pair[i] * pair[j];
                }
            }
        }
        return grad;
    }

    private static double[][] computeFs1Grad(double[][] posPairs, double[][] A, int d) {
        return computeFs1Grad(posPairs, d);
    }

    /**
     * 计算不相似约束梯度：∇f_D
     */
    private static double[][] computeFd1Grad(double[][] negPairs, double[][] A, int d) {
        int m = negPairs.length;
        if (m == 0) {
            return new double[d][d];
        }

        double[][] M = new double[m][d];
        double[] dist = new double[m];
        double sumDist = 0.0;

        for (int i = 0; i < m; i++) {
            double dSq = 0.0;
            for (int k = 0; k < d; k++) {
                double ak = 0.0;
                for (int l = 0; l < d; l++) {
                    ak += A[k][l] * negPairs[i][l];
                }
                dSq += negPairs[i][k] * ak;
            }
            dist[i] = Math.sqrt(Math.max(dSq, 1e-10));
            sumDist += dist[i];

            double factor = 0.5 / (dist[i] + 1e-10);
            for (int k = 0; k < d; k++) {
                M[i][k] = factor * negPairs[i][k];
            }
        }

        double[][] grad = new double[d][d];
        for (int i = 0; i < m; i++) {
            double w = 1.0 / (sumDist + 1e-6);
            for (int k = 0; k < d; k++) {
                for (int l = 0; l < d; l++) {
                    grad[k][l] += w * M[i][k] * negPairs[i][l];
                }
            }
        }

        return grad;
    }

    /**
     * 计算不相似约束函数值：f_D = log(Σ sqrt(d_{ij}' A d_{ij}))
     */
    private static double computeFdObj(double[][] negPairs, double[][] A) {
        int m = negPairs.length;
        int d = A.length;
        if (m == 0) {
            return 0.0;
        }

        double sumDist = 0.0;
        for (int i = 0; i < m; i++) {
            double dSq = 0.0;
            for (int k = 0; k < d; k++) {
                for (int l = 0; l < d; l++) {
                    dSq += negPairs[i][k] * A[k][l] * negPairs[i][l];
                }
            }
            sumDist += Math.sqrt(Math.max(dSq, 1e-10));
        }

        return Math.log(sumDist + 1e-6);
    }

    /**
     * 计算 w 向量（所有相似对外积的展平求和）
     */
    private static double[] computeW(double[][] posPairs, int d) {
        double[] w = new double[d * d];
        for (double[] pair : posPairs) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    w[i * d + j] += pair[i] * pair[j];
                }
            }
        }
        return w;
    }

    /**
     * 梯度投影：使 grad1 正交于 grad2（单位化后）
     */
    private static double[][] gradProjection(double[][] grad1, double[][] grad2) {
        int d = grad1.length;
        double normG2 = 0.0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                normG2 += grad2[i][j] * grad2[i][j];
            }
        }
        normG2 = Math.sqrt(Math.max(normG2, 1e-10));
        if (normG2 < 1e-10) {
            return grad1;
        }

        double[][] g2Norm = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                g2Norm[i][j] = grad2[i][j] / normG2;
            }
        }

        // 计算 <grad1, g2Norm>
        double dot = 0.0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                dot += grad1[i][j] * g2Norm[i][j];
            }
        }

        double[][] result = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                result[i][j] = grad1[i][j] - dot * g2Norm[i][j];
            }
        }

        double normResult = 0.0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                normResult += result[i][j] * result[i][j];
            }
        }
        normResult = Math.sqrt(Math.max(normResult, 1e-10));
        if (normResult < 1e-10) {
            return result;
        }

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                result[i][j] /= normResult;
            }
        }

        return result;
    }

    private static double[] flatten(double[][] M, int r, int c) {
        double[] out = new double[r * c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(M[i], 0, out, i * c, c);
        }
        return out;
    }

    private static void copyMatrix(double[][] src, double[][] dst) {
        for (int i = 0; i < src.length; i++) {
            System.arraycopy(src[i], 0, dst[i], 0, src[i].length);
        }
    }
}
