package com.reremouse.lab.math.ml.lr;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

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
    public float predict(IVector x);
    
}
