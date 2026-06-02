package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 嵌入空间上的<strong>成对距离</strong>、<strong>softmax 行分布</strong>及 NCA / MCML 相关的
 * 权重核与损失——供低秩监督度量迭代的内层循环使用。
 *
 * <p>本类为纯静态数值核，<strong>不</strong>实现 {@link ISupervisedDml}；上层如
 * {@link com.yishape.lab.math.ml.dml.nca.NcaDml}、{@link com.yishape.lab.math.ml.dml.mcml.McmlDml}
 * 等在训练循环中调用此处例程。</p>
 *
 * <p>稠密线性块（嵌入、成对平方距离、梯度中的矩阵链）委托 {@link IMatrix}
 *（Double / Float 可混用数组包装），主要矩阵乘走
 * {@link com.yishape.lab.math.compute.DoubleVectorComputer} /
 * {@link com.yishape.lab.math.compute.FloatVectorComputer}。</p>
 *
 * <p>{@code double[][]} / {@code float[][]} 重载面向仍握有原生数组的训练循环；索引参数
 * {@code n,d,r} 须与数组实际形状一致（调用方负责）。</p>
 *
 * <h2>参考文献（软最大邻域与损失）</h2>
 * <ul>
 *   <li>Goldberger, J., Roweis, S. T., Hinton, G. E., &amp; Salakhutdinov, R. (2005). Neighbourhood
 *       components analysis. <em>NeurIPS</em> 17.</li>
 *   <li>Globerson, A., &amp; Roweis, S. T. (2006). Metric learning by collapsing classes.
 *       <em>NeurIPS</em> 18.</li>
 * </ul>
 *
 * @see IMatrix#multiplyByTransposeOf
 * @see IMatrix#pairwiseSquaredRowDistances
 */
public final class MetricEmbeddingOps {

    private MetricEmbeddingOps() {
    }

    /**
     * 行嵌入矩阵上行向量的两两平方欧氏距离：{@code D[i][j] = ‖emb[i]−emb[j]‖²}。
     *
     * @param emb 行主序 {@code n×r} 嵌入
     * @param n   样本数，须等于 {@code emb.length}
     * @param r   嵌入维，须等于 {@code emb[i].length}
     * @return {@code n×n} 对称矩阵
     */
    public static double[][] pairwiseSquaredDistances(double[][] emb, int n, int r) {
        if (n == 0) {
            return new double[0][0];
        }
        return IMatrix.of(emb).pairwiseSquaredRowDistances().toDoubleArray();
    }

    /**
     * 单精度版本；语义同 {@link #pairwiseSquaredDistances(double[][], int, int)}。
     *
     * @param emb 行主序 {@code n×r}
     * @param n   样本数
     * @param r   嵌入维
     * @return {@code n×n} 的 {@code float} 矩阵
     */
    public static float[][] pairwiseSquaredDistances(float[][] emb, int n, int r) {
        if (n == 0) {
            return new float[0][0];
        }
        return IMatrix.of(emb).pairwiseSquaredRowDistances().toFloatArray();
    }

    /**
     * 线性嵌入：若 {@code X} 为 {@code n×d}、{@code L} 为 {@code r×d}（行即嵌入坐标轴），
     * 则 {@code E = X Lᵀ}，{@code E[i]} 为第 {@code i} 行样本的 {@code r} 维嵌入。
     *
     * @param x 行样本 {@code n×d}
     * @param n 行数（须兼容 {@code x}）
     * @param d 列数
     * @param l 左乘因子 {@code r×d}
     * @param r 输出嵌入维（须兼容 {@code l} 行数）
     * @return {@code n×r}
     */
    public static double[][] embed(double[][] x, int n, int d, double[][] l, int r) {
        return IMatrix.of(x).multiplyByTransposeOf(IMatrix.of(l)).toDoubleArray();
    }

    /**
     * 单精度 {@link #embed(double[][], int, int, double[][], int)}。
     */
    public static float[][] embed(float[][] x, int n, int d, float[][] l, int r) {
        return IMatrix.of(x).multiplyByTransposeOf(IMatrix.of(l)).toFloatArray();
    }

    /**
     * 对每个行索引 {@code i}，在 {@code j≠i} 上计算条件分布
     * {@code p(j|i) ∝ exp(−distSq[i][j])}；对角 {@code pOut[i][i]=0}。
     * 内部采用行内 log-sum-exp 稳定化。
     *
     * @param distSq 平方距离矩阵（通常非负）
     * @param n      阶数
     * @param pOut   与 {@code distSq} 同形的输出缓冲，将被覆盖写入
     */
    public static void softmaxConditionalFromNegSqDist(double[][] distSq, int n, double[][] pOut) {
        for (int i = 0; i < n; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                double v = -distSq[i][j];
                if (v > max) {
                    max = v;
                }
            }
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                sum += Math.exp(-distSq[i][j] - max);
            }
            double logZ = Math.log(sum) + max;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    pOut[i][j] = 0.0;
                } else {
                    pOut[i][j] = Math.exp(-distSq[i][j] - logZ);
                }
            }
        }
    }

    /**
     * NCA 用加权核：对同类掩码与 softmax 邻域分布的组合梯度项（对齐 metric-learn / sklearn 形式）。
     *
     * @param p          条件概率矩阵 {@code p(j|i)}
     * @param sameLabel  {@code sameLabel[i][j]} 当且仅当 {@code i,j} 同类
     * @param n          阶数
     * @param weighted   输出缓冲 {@code n×n}
     */
    public static void ncaWeightedKernel(double[][] p, boolean[][] sameLabel, int n, double[][] weighted) {
        for (int i = 0; i < n; i++) {
            double pi = 0.0;
            for (int j = 0; j < n; j++) {
                if (sameLabel[i][j]) {
                    pi += p[i][j];
                }
            }
            for (int j = 0; j < n; j++) {
                double masked = sameLabel[i][j] ? p[i][j] : 0.0;
                weighted[i][j] = masked - p[i][j] * pi;
            }
        }
    }

    /**
     * MCML 目标条件分布：对每个 {@code i}，在同类 {@code j≠i} 上均匀，其它位置为 0；
     * 若无同类近邻则该行全零。
     *
     * @param y 类别索引
     * @param n 阶数
     * @param q 输出 {@code n×n} 目标分布
     */
    public static void mcmlTargetConditional(int[] y, int n, double[][] q) {
        int[] sameCnt = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i && y[j] == y[i]) {
                    sameCnt[i]++;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i && y[j] == y[i] && sameCnt[i] > 0) {
                    q[i][j] = 1.0 / sameCnt[i];
                } else {
                    q[i][j] = 0.0;
                }
            }
        }
    }

    /**
     * MCML 梯度用加权差 {@code q − p}（与交叉熵对 logit 的梯度符号一致）。
     */
    public static void mcmlWeightedKernel(double[][] p, double[][] q, int n, double[][] weighted) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                weighted[i][j] = q[i][j] - p[i][j];
            }
        }
    }

    /**
     * 将加权矩阵对称化并校正行和：{@code sym = W + Wᵀ}，再令对角为负的列和，与 NCA/MCML 的散度型梯度一致。
     *
     * @param weighted 输入方阵
     * @param n        阶数
     * @param symOut   输出缓冲
     */
    public static void skewSymmetrizeForGradient(double[][] weighted, int n, double[][] symOut) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                symOut[i][j] = weighted[i][j] + weighted[j][i];
            }
        }
        double[] colSum = new double[n];
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += weighted[k][j];
            }
            colSum[j] = s;
        }
        for (int i = 0; i < n; i++) {
            symOut[i][i] = -colSum[i];
        }
    }

    /**
     * 关于嵌入矩阵 {@code L} 的梯度链：{@code ∇L = 2 Eᵀ S X}，其中 {@code E} 为 {@code n×r} 嵌入、
     * {@code S} 为 {@code symOut}、{@code X} 为原始特征 {@code n×d}；结果写入 {@code gradL}（{@code r×d}）。
     */
    public static void gradientLinearTransform(double[][] emb, double[][] sym, double[][] x, int n, int d, int r,
            double[][] gradL) {
        IMatrix<Double> grad = IMatrix.of(emb).transpose()
                .mmul(IMatrix.of(sym).mmul(IMatrix.of(x)))
                .mmul(2.0);
        double[][] gd = grad.toDoubleArray();
        for (int p = 0; p < r; p++) {
            System.arraycopy(gd[p], 0, gradL[p], 0, d);
        }
    }

    /**
     * 单精度 {@link #gradientLinearTransform(double[][], double[][], double[][], int, int, int, double[][])}。
     */
    public static void gradientLinearTransform(float[][] emb, float[][] sym, float[][] x, int n, int d, int r,
            float[][] gradL) {
        IMatrix<Float> grad = IMatrix.of(emb).transpose()
                .mmul(IMatrix.of(sym).mmul(IMatrix.of(x)))
                .multiplyByScalar(2.0f);
        float[][] gf = grad.toFloatArray();
        for (int p = 0; p < r; p++) {
            System.arraycopy(gf[p], 0, gradL[p], 0, d);
        }
    }

    /**
     * NCA 辅助量：对同类对 {@code (i,j)} 的 {@code p[i][j]} 求和（与测试及 leave-one-out 吸引同类邻域目标一致；值越大表示同类分配越多）。
     *
     * @param p          softmax 条件分布
     * @param sameLabel 同类指示
     * @param n          阶数
     * @return 标量和
     */
    public static double ncaLoss(double[][] p, boolean[][] sameLabel, int n) {
        double loss = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (sameLabel[i][j]) {
                    loss += p[i][j];
                }
            }
        }
        return loss;
    }

    /**
     * MCML：目标分布 {@code q} 与模型分布 {@code p} 之间的 KL（略去与 {@code p} 无关的常数项形式）。
     */
    public static double mcmlKlLoss(double[][] p, double[][] q, int n) {
        double loss = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i && q[i][j] > 0.0) {
                    loss += q[i][j] * (Math.log(q[i][j]) - Math.log(Math.max(p[i][j], 1e-20)));
                }
            }
        }
        return loss;
    }
}
