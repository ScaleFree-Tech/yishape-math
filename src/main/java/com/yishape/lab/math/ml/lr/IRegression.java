package com.yishape.lab.math.ml.lr;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;

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
    public RegressionResult fit(IMatrix feature, IVector labels);
    
    /**
     * 
     * @param x
     * @return 
     */
    public double predict(IVector x);
    
}
