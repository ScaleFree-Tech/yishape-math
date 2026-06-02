package com.yishape.lab.math.ml.reg;


import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.IRichReport;
import com.yishape.lab.util.ReportBuilder;

/**
 *
 * @author lteb2
 */
public class RegressionResult implements IRichReport {
    
    private IVector weights;

    private IVector bias;

    private double loss;

    /** 训练集决定系数 R²，由 {@link RereLinearRegression#fit} 写入 */
    private double r2Score;

    /**
     * 训练集 RMSE（均方根误差，{@code sqrt( (1/n) * Σ(y - ŷ)² )}），由 {@link RereLinearRegression#fit} 写入
     */
    private double rmse;

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

    /**
     * 训练集上的 RMSE（与 {@link #getR2Score()} 同一批数据），在 {@link RereLinearRegression#fit} 完成后可用。
     */
    public double getRmse() {
        return rmse;
    }

    void setRmse(double rmse) {
        this.rmse = rmse;
    }

    @Override
    public String toReport() {
        ReportBuilder rb = new ReportBuilder("Linear Regression Result");
        if (weights != null) rb.kv("Weights dim", weights.size());
        if (bias != null) rb.kv("Bias dim", bias.size());
        rb.kv("Loss", String.format("%.6f", loss));
        rb.kv("R-squared", r2Score);
        rb.kv("RMSE", rmse);
        return rb.build();
    }

    @Override
    public String toBriefReport() {
        return String.format("Regression | R2=%.4f | RMSE=%.4f", r2Score, rmse);
    }
}
