package com.yishape.lab.math.ml.metric;

import org.slf4j.LoggerFactory;

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
    public default void log(int k, CrossValidationResult middleResult) {
        var logger = LoggerFactory.getLogger(CrossValidationLogger.class);
        logger.debug("第 {} 折训练完成 / {}-th fold's training finished: ", k, k);
        logger.debug("{}", middleResult);
    }

}
