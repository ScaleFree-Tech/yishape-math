package com.reremouse.lab.math.ml.lr;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
public interface IRegression {
    
    /**
     * 
     * @param feature
     * @param labels
     * @return 
     */
    public RegressionResult fit(IMatrix feature,IVector labels);
    
    /**
     * 
     * @param x
     * @return 
     */
    public double predict(IVector x);
    
}
