package com.yishape.lab.math.ml.dml.triplet;

import com.yishape.lab.math.linalg.IVector;

import java.util.Objects;

/**
 * 对角度量学习（DDML）流水线中的<strong>三元组</strong>样本：锚点 {@code xi}、同类参照 {@code xj}、
 * 异类或对照点 {@code xk}，并携带欧氏空间预计算距离与抽样权重。
 *
 * <p>与仓库内 Julia 参考实现 {@code TripletModule.jl} 的字段语义对齐，供
 * {@link TripletBuilder} 与 {@link com.yishape.lab.math.ml.dml.ddml} 系数构造使用。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Schultz, M., &amp; Joachims, T. (2003). Learning a distance metric from relative comparisons.
 *       In <em>Advances in Neural Information Processing Systems (NeurIPS) 16</em>, pp. 41–48.
 *       （相对约束 / 三元组动机。）</li>
 * </ul>
 */
public final class Triplet {

    private final IVector<Double> xi;
    private final IVector<Double> xj;
    private final IVector<Double> xk;
    private final double ijDis;
    private final double jkDis;
    private double weight;

    /**
     * @param xi    锚点向量（拷贝存储）
     * @param xj    同类端点
     * @param xk    异类或对照端点
     * @param ijDis 预计算的 {@code ‖xi−xj‖²}（或管线约定的一致定义）
     * @param jkDis 预计算的 {@code ‖xj−xk‖²} 相关量（与 LP 系数一致）
     * @param weight 初始抽样权重，可被外部调整
     */
    public Triplet(IVector<Double> xi, IVector<Double> xj, IVector<Double> xk, double ijDis, double jkDis,
            double weight) {
        this.xi = Objects.requireNonNull(xi, "xi").copy();
        this.xj = Objects.requireNonNull(xj, "xj").copy();
        this.xk = Objects.requireNonNull(xk, "xk").copy();
        if (xj.length() != xi.length() || xk.length() != xi.length()) {
            throw new IllegalArgumentException("xi, xj, xk 维数须一致");
        }
        this.ijDis = ijDis;
        this.jkDis = jkDis;
        this.weight = weight;
    }

    public int dimension() {
        return xi.length();
    }

    /** 锚点样本 {@code x^i}（Schultz/Joachims 记号）。 */
    public IVector<Double> xi() {
        return xi;
    }

    /** 与锚点同类的参照 {@code x^j}。 */
    public IVector<Double> xj() {
        return xj;
    }

    /** 异类或对照端点 {@code x^k}。 */
    public IVector<Double> xk() {
        return xk;
    }

    /** 预计算的同类段距离（与 {@link TripletBuilder} 中近邻距离定义一致，一般为平方欧氏）。 */
    public double ijDis() {
        return ijDis;
    }

    /** 参照点 {@code x^j} 到异类点 {@code x^k} 段的预计算距离。 */
    public double jkDis() {
        return jkDis;
    }

    /** 抽样权重；可经 {@link #setWeight(double)} 调整。 */
    public double weight() {
        return weight;
    }

    /** 改写抽样权重（如重加权或归一化）。 */
    public void setWeight(double weight) {
        this.weight = weight;
    }

    /** DDML 参考实现中的默认温度参数 {@code rou}。 */
    public static double defaultRou() {
        return 1.0 / 4.5;
    }

    /**
     * 指数型距离间隔权重：间隔越大权重越小。
     *
     * @param ijDis 同类段距离量
     * @param jkDis 跨越异类段距离量
     * @param rou   温度，常用 {@link #defaultRou()}
     */
    public static double computeWeight(double ijDis, double jkDis, double rou) {
        double gap = Math.abs(ijDis - jkDis);
        return Math.exp(-gap / rou);
    }
}
