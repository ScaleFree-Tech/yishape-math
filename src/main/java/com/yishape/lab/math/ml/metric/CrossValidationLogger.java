package com.yishape.lab.math.ml.metric;

import org.slf4j.LoggerFactory;

/**
 * 本类用于k折交叉检验时完成某折时的日志输出
 *
 * @author lteb2
 */
public interface CrossValidationLogger {

    /**
     * 输出日志
     *
     * @param k
     * @param middleResult
     */
    public default void log(int k, CrossValidationResult middleResult) {
        var logger = LoggerFactory.getLogger(CrossValidationLogger.class);
        logger.debug("第 {} 折训练完成 / {}-th fold's training finished: ", k, k);
        logger.debug("{}", middleResult);
    }

}
