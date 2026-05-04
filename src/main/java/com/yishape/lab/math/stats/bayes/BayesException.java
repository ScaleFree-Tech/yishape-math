package com.yishape.lab.math.stats.bayes;

/**
 * 贝叶斯分析相关的异常类 / Exception class for Bayesian analysis related errors
 * <p>
 * 当贝叶斯分析过程中发生错误时抛出此异常。
 * Thrown when an error occurs during Bayesian analysis.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class BayesException extends RuntimeException {
    
    /**
     * 创建贝叶斯异常 / Create a BayesException
     *
     * @param message 异常消息 / Exception message
     */
    public BayesException(String message) {
        super(message);
    }

    /**
     * 创建贝叶斯异常（带原因） / Create a BayesException with cause
     *
     * @param message 异常消息 / Exception message
     * @param cause 异常原因 / Cause of the exception
     */
    public BayesException(String message, Throwable cause) {
        super(message, cause);
    }
}