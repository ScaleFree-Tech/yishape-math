package com.yishape.lab.math.optimize.regularization;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

import java.util.Objects;

/**
 * L2（平方）正则项，与 {@code RereDiagDmlADMMDistributed.jl} 中
 * {@code compute_reg_value} / {@code compute_reg_grad} 的纯 L2 部分一致。
 * <p>
 * 目标为各分量平方和 {@code ‖x‖₂² = Σ x_i²}（不是 {@code ½‖x‖²}）；梯度为 {@code ∇ = 2x}。
 * 若损失里写作 {@code λ‖x‖²}，等价于本类目标乘 {@code λ}，梯度乘 {@code λ}。
 * </p>
 * <p>
 * <strong>权重：</strong>不写死 {@code (1-alpha)} 或 {@code regWeight}，由调用方与 L1、数据项等组合。
 * </p>
 *
 * @author lteb2
 */
@SuppressWarnings("rawtypes")
public class RereL2Regularization implements IGradientFunction, IObjectiveFunction {

    /**
     * 梯度 {@code ∇(Σ x_i²) = 2x}，逐分量计算后封装为 {@link IVector}。
     *
     * @param x 变量向量
     * @return 同维梯度向量
     */
    @Override
    public IVector computeGradient(IVector x) {
        Objects.requireNonNull(x, "x");
        int n = x.length();
        double[] gra = new double[n];
        for (int i = 0; i < n; i++) {
            // ∂(x_i²)/∂x_i = 2 x_i
            gra[i] = 2.0 * x.get(i);
        }
        return Linalg.vector(gra);
    }

    /**
     * {@code f(x) = Σ x_i²}，即向量欧氏范数的平方（未开方）。
     *
     * @param x 变量向量
     * @return 标量 {@code ‖x‖₂²}
     */
    @Override
    public double computeObjective(IVector x) {
        Objects.requireNonNull(x, "x");
        double sum = 0.0;
        for (int i = 0; i < x.length(); i++) {
            double z = x.get(i);
            sum += z * z; // 累加 x_i²
        }
        return sum;
    }
}
