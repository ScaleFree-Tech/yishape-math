package com.reremouse.lab.audio.embedding;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
public interface IAudioEmbedding {
    
    /**
     * 基于MFCC特征生成定长的向量表征
     * @param mfcc
     * @return 
     */
    public IVector embed(IMatrix<Double> mfcc);
    
}
