package com.reremouse.lab.math.ml.cls;


import com.reremouse.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
public class LogisticRegressionResult extends ClassificationResult {

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
