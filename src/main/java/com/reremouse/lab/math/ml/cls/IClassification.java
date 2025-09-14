package com.reremouse.lab.math.ml.cls;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
public interface IClassification {
    
    /**
     * 
     * @param feature
     * @param labels
     * @return 
     */
    public ClassificationResult fit(IMatrix feature,String[] labels);
    
    /**
     * 
     * @param x
     * @return 
     */
    public String predict(IVector x);
    
}
