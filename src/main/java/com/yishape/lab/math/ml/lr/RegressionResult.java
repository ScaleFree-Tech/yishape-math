package com.yishape.lab.math.ml.lr;


import com.yishape.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
public class RegressionResult {
    
    private IVector weights;

    private IVector bias;

    private double loss;

    /** 训练集决定系数 R²，由 {@link RereLinearRegression#fit} 写入 */
    private double r2Score;

    public IVector getWeights() {
        return weights;
    }

    public void setWeights(IVector weights) {
        this.weights = weights;
    }

    public IVector getBias() {
        return bias;
    }

    public void setBias(IVector bias) {
        this.bias = bias;
    }

    public double getLoss() {
        return loss;
    }

    public void setLoss(double loss) {
        this.loss = loss;
    }

    /**
     * 训练集上的决定系数 R²（{@code 1 - SS_res/SS_tot}），在 {@link RereLinearRegression#fit} 完成后可用。
     */
    public double getR2Score() {
        return r2Score;
    }

    void setR2Score(double r2Score) {
        this.r2Score = r2Score;
    }
}
