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
    
    
}
