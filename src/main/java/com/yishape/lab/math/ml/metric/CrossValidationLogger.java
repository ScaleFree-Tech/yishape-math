package com.yishape.lab.math.ml.metric;

import com.yishape.lab.util.YishapeLogger;

/**
 * 交叉验证日志记录器接口 / Cross Validation Logger Interface
 * <p>
 * 定义在K折交叉验证过程中记录日志的接口。
 * Defines the interface for logging during K-fold cross validation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface CrossValidationLogger {

    YishapeLogger CV_LOG = YishapeLogger.getLogger(CrossValidationLogger.class);

    /**
     * 输出交叉验证日志 / Log Cross Validation Result
     * <p>
     * 在每折交叉验证完成后输出日志。
     * Log after each fold of cross validation is completed.
     * </p>
     *
     * @param k 折编号 / Fold number
     * @param middleResult 交叉验证结果 / Cross validation result
     */
    default void log(int k, CrossValidationResult middleResult) {
        if (CV_LOG.isDebugEnabled()) {
            CV_LOG.debug("Fold {} training completed: {}", k, middleResult);
        }
    }

}
