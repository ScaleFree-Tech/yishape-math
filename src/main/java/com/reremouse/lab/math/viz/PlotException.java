package com.reremouse.lab.math.viz;

/**
 * 绘图异常类
 * 用于处理绘图过程中的各种异常情况
 * @author lteb2
 */
public class PlotException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * 构造函数
     * @param message 异常消息
     */
    public PlotException(String message) {
        super(message);
    }
    
    /**
     * 构造函数
     * @param message 异常消息
     * @param cause 原因异常
     */
    public PlotException(String message, Throwable cause) {
        super(message, cause);
    }
    
    /**
     * 构造函数
     * @param cause 原因异常
     */
    public PlotException(Throwable cause) {
        super(cause);
    }
}
