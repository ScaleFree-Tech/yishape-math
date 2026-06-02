package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import java.io.Serializable;

/**
 * 非监督型距离度量学习的统一<strong>拟合契约</strong>：实现者保存算法族与超参数，
 * {@link #fit} 仅消费特征，产出 {@link DmlMetric}。
 * 
 * @author lteb2
 */
public interface IUnsupervisedDml extends Serializable{
    
    /**
     * 从数值特征拟合度量。
     *
     * @param features 行样本，{@link IMatrix#getColNum()} 为输入维
     * @return 非 null 的拟合结果
     */
    DmlMetric fit(IMatrix<Double> features);
    
}
