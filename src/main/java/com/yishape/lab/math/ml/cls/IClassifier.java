package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import java.util.Map;

/**
 *
 * @author lteb2
 */
public interface IClassifier extends ISerializableModel{
    
    
    /**
     * 
     * @param feature
     * @param labels
     * @return 
     */
    public ClassificationResult fit(IMatrix feature, String[] labels);
    
    
    /**
     * 预测类别
     * @param x
     * @return 
     */
    public String predict(IVector x);
    
    /**
     * 预测各个类别的概率
     * @param x
     * @return 
     */
    public Map<String, Double> predictProb(IVector x);
    
    /**
     * 批量预测类别
     * @param features
     * @return 
     */
    public String[] predictBatch(IMatrix features);
    
    
    /**
     * 
     * @param features
     * @return 
     */
    public BatchPredictionResult predictBatchWithProbs(IMatrix features);
    
    
    /**
     * 
     * @return 
     */
    public boolean isTrained();
    
    /**
     * 
     * @return 
     */
    public ClassificationMetrics getMetrics();
    
    /**
     * 
     * @param metrics
     */
    public void setMetrics(ClassificationMetrics metrics);
    
}
