package com.yishape.lab.math.optimize.regularization;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

import java.util.Objects;

/**
 * L1 正则项的光滑近似（与 {@code refs/ddml/julia_src/solver/RereDiagDmlADMMDistributed.jl} 中
 * {@code sign_for_l1} / {@code sign_for_l1_gradient} 对齐：Huber 型，平滑参数 {@code param_c = 1e-10}）。
 * <p>
 * 对每个分量 {@code z}，用分段函数逼近 {@code |z|}，在 {@code |z| &lt; param_c} 内用二次函数保证原点处可微；
 * {@code |z| ≥ param_c} 时为绝对值的左右线性支。值为各分量之和；梯度按分段求导。
 * </p>
 * <p>
 * <strong>权重：</strong>本类<strong>不</strong>乘 DDML 中的 {@code alpha} 或 {@code regWeight}，
 * 与 L2 组合或 ADMM 时由外层按 {@code alpha * L1 + (1-alpha) * L2} 等形式自行缩放。
 * </p>
 *
 * @author lteb2
 */
@SuppressWarnings("rawtypes")
public class RereL1Regularization implements IGradientFunction, IObjectiveFunction {

    /**
     * 光滑宽度；与 Julia 中 {@code param_c} 一致。越小越接近真 L1，但零点邻域梯度变化越陡。
     */
    private static final double PARAM_C = 1.0e-10;

    /**
     * 光滑 L1 的梯度，分量形式：
     * {@code |z| &lt; c ⇒ z/c}；{@code z ≥ c ⇒ 1}；{@code z ≤ -c ⇒ -1}。
     *
     * @param x 变量向量（与优化问题中记号 z 对应）
     * @return 与 x 同长度的梯度向量（元素为 double）
     */
    @Override
    public IVector computeGradient(IVector x) {
        Objects.requireNonNull(x, "x");
        int n = x.length();
        double[] gra = new double[n];
        double c = PARAM_C;
        for (int i = 0; i < n; i++) {
            double z = x.get(i);
            double az = Math.abs(z);
            if (az < c) {
                // d/dz [ z²/(2c) + c/2 ] = z/c
                gra[i] = z / c;
            } else if (z >= c) {
                gra[i] = 1.0;
            } else {
                // z <= -c
                gra[i] = -1.0;
            }
        }
        return Linalg.vector(gra);
    }

    /**
     * 光滑 L1 惩罚值 {@code Σ φ(z_i)}，其中 {@code φ} 为上述 Huber 型逼近，在远场等价于 {@code |z|} 之和。
     *
     * @param x 变量向量
     * @return 标量目标值（各分量贡献之和）
     */
    @Override
    public double computeObjective(IVector x) {
        Objects.requireNonNull(x, "x");
        double sum = 0.0;
        double c = PARAM_C;
        double twoC = 2.0 * c;
        for (int i = 0; i < x.length(); i++) {
            double z = x.get(i);
            double az = Math.abs(z);
            if (az < c) {
                // 二次段：与两端 |z| 在 z=±c 处 C¹ 衔接
                sum += z * z / twoC + c / 2.0;
            } else if (z >= c) {
                sum += z; // 正半轴：|z| = z
            } else {
                sum += -z; // z <= -c：|z| = -z
            }
        }
        return sum;
    }
}
