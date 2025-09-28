package com.reremouse.lab.math.stats.bayes;

/**
 * 贝叶斯分析相关的异常类
 * Exception class for Bayesian analysis related errors
 */
public class BayesException extends RuntimeException {
    
    public BayesException(String message) {
        super(message);
    }
    
    public BayesException(String message, Throwable cause) {
        super(message, cause);
    }
}