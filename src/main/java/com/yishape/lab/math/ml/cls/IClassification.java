package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;

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
    public ClassificationResult fit(IMatrix feature, String[] labels);
    
    /**
     * 
     * @param x
     * @return 
     */
    public String predict(IVector x);
    
}
