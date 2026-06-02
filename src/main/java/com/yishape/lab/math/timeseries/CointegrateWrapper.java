package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

/**
 * 协整分析包装器 / Cointegration Analysis Wrapper.
 * 提供统一的协整分析入口。
 */
public class CointegrateWrapper {

    public CointegrationAnalysis.EngleGrangerResult engleGrangerTest(IVector<Double> y, IVector<Double> x, int maxLags) {
        return CointegrationAnalysis.engleGrangerTest(y, x, maxLags);
    }

    public CointegrationAnalysis.JohansenResult johansenTest(IMatrix<Double> data, int maxLags, CointegrationAnalysis.TrendType trendType) {
        return CointegrationAnalysis.johansenTest(data, maxLags, trendType);
    }

    public CointegrationAnalysis.CointegratingRelationship estimateCointegratingRelationship(IVector<Double> y, IVector<Double> x) {
        return CointegrationAnalysis.estimateCointegratingRelationship(y, x);
    }

    public CointegrationAnalysis.ErrorCorrectionModel estimateECM(IVector<Double> deltaY, IVector<Double> deltaX,
                                                                    IVector<Double> residuals, int maxLags) {
        return CointegrationAnalysis.estimateECM(deltaY, deltaX, residuals, maxLags);
    }
}
