package com.reremouse.lab.math.ml.cls;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.ml.ISerializableModel;

/**
 *
 * @author lteb2
 */
public interface IClassification extends ISerializableModel{
    
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
