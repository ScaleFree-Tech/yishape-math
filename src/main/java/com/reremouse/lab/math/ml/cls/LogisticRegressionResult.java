package com.reremouse.lab.math.ml.cls;


import com.reremouse.lab.math.IVector;

/**
 *
 * @author lteb2
 */
public class LogisticRegressionResult extends ClassificationResult {

    private IVector weights;

    private IVector bias;

    private float loss;

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

    public float getLoss() {
        return loss;
    }

    public void setLoss(float loss) {
        this.loss = loss;
    }

    
    
    
}
