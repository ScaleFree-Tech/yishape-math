package com.yishape.lab.math.ml.dml.ddml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.triplet.Triplet;

import java.util.Arrays;
import java.util.List;

/**
 * 由相对约束三元组构造对角 DDML 的线性规划标准型系数 {@code (A, b, c)}，对应 Julia
 * {@code refs/ddml/julia_src/rere_dml/DiagDml.jl} 中的 {@code create_coefficients_with_triplets}。
 *
 * <p><strong>决策变量</strong>（长度 {@code m + 2n}，与 Julia {@code cat} 拼接顺序一致）：</p>
 * <ul>
 *   <li>前 {@code m} 个：各特征维上的对角度量参数（平方根空间中的非负权重，最终 {@link RereDiagDml} 再 {@code sqrt}）；</li>
 *   <li>中间 {@code n} 个：每条三元组约束的<strong>松弛</strong>变量（系数块为单位阵 {@code I}）；</li>
 *   <li>后 {@code n} 个：<strong>剩余</strong>变量（系数块为 {@code -I}），用于将不等式约束写成 {@code A x &gt;= b} 的可行形式。</li>
 * </ul>
 *
 * <p><strong>第 {@code i} 行约束</strong>（对应第 {@code i} 个三元组）：对前 {@code m} 列，系数为
 * {@code (x^k-x^j)² - (x^i-x^j)²} 按维；在松弛列 {@code m+i} 为 {@code 1}，在剩余列 {@code m+n+i} 为 {@code -1}。
 * 形如「加权距离差」≤ 松弛 − 剩余，与 Schultz/Joachims 型相对距离约束一致。右端 {@code b_i} 一般为 {@code tau}（非 huber 时乘三元组权重）。</p>
 *
 * <p><strong>目标线性项 {@code c}</strong>：前 {@code m} 维来自同类近邻与异类 impostor 的 huber 或平方距离差（见实现）；
 * 松弛段每维为 {@code punishmentMu}；剩余段为 {@code 0}。</p>
 *
 * <p>{@link RereDiagDmlSolverLangMul} 不需要剩余变量列时，调用 {@link #truncateForLangMul()}，与 Julia
 * {@code c[1:m+n], A[:,1:m+n]} 截断一致。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Schultz, M., &amp; Joachims, T. (2003). Learning a distance metric from relative comparisons.
 *       <em>NeurIPS</em> 16.</li>
 *   <li>Rosales, R., &amp; Fung, G. (2006). Learning sparse metrics via linear programming. <em>ICML</em>.</li>
 * </ul>
 */
public final class DiagDmlCoefficients {

    /** 行：{@code n} 个三元组约束；列：{@code m + 2n}。 */
    private final double[][] a;
    /** 约束右端，长度 {@code n}。 */
    private final double[] b;
    /** 目标线性系数，长度 {@code m + 2n}。 */
    private final double[] c;
    /** 特征维数 {@code m}。 */
    private final int m;
    /** 三元组条数 {@code n}。 */
    private final int n;

    private DiagDmlCoefficients(double[][] a, double[] b, double[] c, int m, int n) {
        this.a = a;
        this.b = b;
        this.c = c;
        this.m = m;
        this.n = n;
    }

    public int featureDim() {
        return m;
    }

    public int numTriplets() {
        return n;
    }

    public double[][] getA() {
        return a;
    }

    public double[] getB() {
        return b;
    }

    public double[] getC() {
        return c;
    }

    public IMatrix<Double> matrixA() {
        return Linalg.matrix(a);
    }

    public IVector<Double> vectorB() {
        return Linalg.vector(b);
    }

    public IVector<Double> vectorC() {
        return Linalg.vector(c);
    }

    public int variableDim() {
        return c.length;
    }

    /**
     * 截去「剩余变量」列与 {@code c} 尾段，得到仅含特征维 + 松弛变量的子问题；供
     * {@link RereDiagDmlSolverLangMul} 使用，与 {@code DiagDml.jl} 中
     * {@code A[:,1:m+n], c[1:m+n]} 一致。
     *
     * @return 降维后的 {@code (cReduced, aReduced, b)}，维数分别为 {@code m+n}、{@code n×(m+n)}、{@code n}
     */
    public TruncatedLangMulProblem truncateForLangMul() {
        int dim = m + n;
        double[] cR = Arrays.copyOf(c, dim);
        double[][] aR = new double[n][dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, aR[i], 0, dim);
        }
        return new TruncatedLangMulProblem(cR, aR, Arrays.copyOf(b, n));
    }

    /**
     * LangMul 路径所用截断 LP 数据（特征 + 松弛，无剩余变量列）。
     *
     * @param cReduced 长度 {@code m+n}
     * @param aReduced {@code n×(m+n)}
     * @param b        长度 {@code n}
     */
    public record TruncatedLangMulProblem(double[] cReduced, double[][] aReduced, double[] b) {

        public IVector<Double> cVector() {
            return Linalg.vector(cReduced);
        }

        public IMatrix<Double> aMatrix() {
            return Linalg.matrix(aReduced);
        }

        public IVector<Double> bVector() {
            return Linalg.vector(b);
        }
    }

    /**
     * 由三元组列表构造完整 LP 系数；与 {@code DiagDml.jl#create_coefficients_with_triplets} 数值逻辑一致。
     *
     * @param triplets    非空；每条含 {@code xi,xj,xk} 及权重，维数须一致
     * @param punishmentMu 松弛变量在目标中的线性惩罚（Julia 默认 {@code 1e4} / {@code 5000} 量级，{@link RereDiagDml} 默认 5000）
     * @param distance    {@code "huber"}：{@code c} 的头 {@code m} 维用 Huber 平滑平方损失累加；{@code "no_huber"} 或其它：
     *                    与 Julia {@code "not_huber"} 分支一致，对 {@code (xi-xj)²} 与约束行按三元组 {@code weight} 加权
     * @param tau         约束右端基准；非 huber 时 {@code b_i = tau * weight}
     * @return 封装后的 {@code A,b,c} 及维数 {@code m,n}
     */
    public static DiagDmlCoefficients fromTriplets(List<Triplet> triplets, double punishmentMu,
            String distance, double tau) {
        if (triplets == null || triplets.isEmpty()) {
            throw new IllegalArgumentException("triplets 不能为空");
        }
        String dist = distance == null ? "huber" : distance;
        int n = triplets.size();
        int m = triplets.get(0).dimension();
        int dim = m + 2 * n;
        double[][] rawA = new double[n][m];
        double[] cHead = new double[m];
        double[] b = new double[n];
        Arrays.fill(b, tau);

        boolean huber = dist.equalsIgnoreCase("huber");

        for (int i = 0; i < n; i++) {
            Triplet t = triplets.get(i);
            if (t.dimension() != m) {
                throw new IllegalArgumentException("三元组特征维数须一致");
            }
            double w = t.weight();
            IVector<Double> xi = t.xi();
            IVector<Double> xj = t.xj();
            IVector<Double> xk = t.xk();

            if (huber) {
                // DiagDml.jl: thres=0.1，c 累加 Huber 型代理，A 行为 (xk-xj)²-(xi-xj)²（不带样本权重）
                double thres = 0.1;
                for (int j = 0; j < m; j++) {
                    double e = xi.get(j) - xj.get(j);
                    cHead[j] += Math.abs(e) < thres ? (e * e / 2.0) : (Math.abs(e) * thres - thres * thres / 2.0);
                    double djk = xk.get(j) - xj.get(j);
                    double dij = xi.get(j) - xj.get(j);
                    rawA[i][j] = djk * djk - dij * dij;
                }
            } else {
                // DiagDml.jl: distance != "huber"，等价 "not_huber"：行与 c 前段乘 weight，b_i 同步乘权
                for (int j = 0; j < m; j++) {
                    double dij = xi.get(j) - xj.get(j);
                    cHead[j] += dij * dij * w;
                    double djk = xk.get(j) - xj.get(j);
                    rawA[i][j] = ((djk * djk) - (dij * dij)) * w;
                }
                b[i] = tau * w;
            }
        }

        // 拼接松弛 I 与剩余 -I：A 变 [ rawA | I | -I ]，与 Julia cat(A, Matrix(I), -Matrix(I))
        double[][] a = new double[n][dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rawA[i], 0, a[i], 0, m);
            a[i][m + i] = 1.0;
            a[i][m + n + i] = -1.0;
        }

        // c = [ cHead | mu·1_n | 0_n ]，与 Julia cat(c, ones(n).*mu, zeros(n))
        double[] c = new double[dim];
        System.arraycopy(cHead, 0, c, 0, m);
        for (int j = m; j < m + n; j++) {
            c[j] = punishmentMu;
        }
        Arrays.fill(c, m + n, dim, 0.0);

        return new DiagDmlCoefficients(a, b, c, m, n);
    }
}
