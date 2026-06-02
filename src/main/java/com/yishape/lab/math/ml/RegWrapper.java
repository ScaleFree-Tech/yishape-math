package com.yishape.lab.math.ml;

import com.yishape.lab.math.ml.reg.IRegression;
import com.yishape.lab.math.ml.reg.RereLinearRegression;
import com.yishape.lab.util.YishapeLogger;

/**
 *
 * @author lteb2
 */
public class RegWrapper {
    
    
    
    // ========== 回归 / Regression ==========

    /**
     * 普通最小二乘线性回归（含截距、无正则化），等价于 {@code new RereLinearRegression()}。
     *
     * @return 
     * @see RereLinearRegression
     */
    public IRegression linear() {
        return new RereLinearRegression();
    }

    /**
     * 线性回归（含截距），根据 {@code lambda1}、{@code lambda2} 自动选择无正则 / L1 / L2 / ElasticNet，
     * 等价于 {@code new RereLinearRegression(true, lambda1, lambda2)}。
     *
     * @param lambda1 L1 系数（Lasso）
     * @param lambda2 L2 系数（Ridge）
     * @return 
     * @see RereLinearRegression
     */
    public IRegression linear(double lambda1, double lambda2) {
        return new RereLinearRegression(true, lambda1, lambda2);
    }

    /**
     * 线性回归（可关闭截距），正则类型由 {@code lambda1}、{@code lambda2} 推断。
     *
     * @param includeBias 是否包含偏置项
     * @param lambda1     L1 系数
     * @param lambda2     L2 系数
     * @return 
     * @see RereLinearRegression
     */
    public IRegression linear(boolean includeBias, double lambda1, double lambda2) {
        return new RereLinearRegression(includeBias, lambda1, lambda2);
    }

    // ========== 持久化 / Persistence ==========

    /**
     * 从 JSON 文件加载回归模型。
     */
    public static IRegression loadRegression(String modelPath) {
        ISerializableModel model = ISerializableModel.load(modelPath);
        if (model instanceof IRegression reg) {
            return reg;
        }
        throw new IllegalStateException("Loaded model is not an IRegression: " + modelPath);
    }

    /**
     * 将回归模型保存为 JSON 文件。
     */
    public static void saveRegression(IRegression regression, String modelPath) {
        regression.save(modelPath);
    }
}
