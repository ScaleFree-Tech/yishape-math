package com.reremouse.lab.math.ml.lr;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.ml.ISerializableModel;

/**
 *
 * @author lteb2
 */
public interface IRegression extends ISerializableModel{
    
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
