package com.yishape.lab.math.signal.core;

/**
 * 信号处理异常类 / Signal Processing Exception Class
 * <p>
 * 信号处理过程中发生的异常的基础类。
 * Base class for exceptions that occur during signal processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalProcessingException extends Exception {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     */
    public SignalProcessingException(String message) {
        super(message);
    }
    
    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     * @param cause 异常原因 / Exception cause
     */
    public SignalProcessingException(String message, Throwable cause) {
        super(message, cause);
    }
    
    /**
     * 构造函数 / Constructor
     *
     * @param cause 异常原因 / Exception cause
     */
    public SignalProcessingException(Throwable cause) {
        super(cause);
    }
}