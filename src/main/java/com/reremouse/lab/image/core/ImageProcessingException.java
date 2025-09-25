package com.reremouse.lab.image.core;

/**
 * 图像处理异常类 / Image Processing Exception Class
 * <p>
 * 用于处理图像处理过程中发生的各种异常情况。
 * 提供详细的错误信息和异常类型分类。
 * </p>
 * <p>
 * Used to handle various exception conditions during image processing.
 * Provides detailed error information and exception type classification.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class ImageProcessingException extends Exception {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * 异常类型枚举 / Exception Type Enum
     */
    public enum ErrorType {
        INVALID_INPUT,          // 无效输入 / Invalid input
        INVALID_PARAMETERS,     // 无效参数 / Invalid parameters
        PROCESSING_FAILED,      // 处理失败 / Processing failed
        INSUFFICIENT_MEMORY,    // 内存不足 / Insufficient memory
        ALGORITHM_ERROR,        // 算法错误 / Algorithm error
        IO_ERROR,              // IO错误 / IO error
        UNSUPPORTED_OPERATION,  // 不支持的操作 / Unsupported operation
        TIMEOUT,               // 超时 / Timeout
        GPU_ERROR,             // GPU错误 / GPU error
        UNKNOWN                // 未知错误 / Unknown error
    }
    
    private ErrorType errorType;
    private String detailMessage;
    private String processorName;
    
    /**
     * 构造函数 / Constructor
     * 
     * @param message 错误消息 / Error message
     */
    public ImageProcessingException(String message) {
        super(message);
        this.errorType = ErrorType.UNKNOWN;
        this.detailMessage = message;
    }
    
    /**
     * 构造函数 / Constructor
     * 
     * @param message 错误消息 / Error message
     * @param cause 原因 / Cause
     */
    public ImageProcessingException(String message, Throwable cause) {
        super(message, cause);
        this.errorType = ErrorType.UNKNOWN;
        this.detailMessage = message;
    }
    
    /**
     * 构造函数 / Constructor
     * 
     * @param errorType 错误类型 / Error type
     * @param message 错误消息 / Error message
     */
    public ImageProcessingException(ErrorType errorType, String message) {
        super(message);
        this.errorType = errorType;
        this.detailMessage = message;
    }
    
    /**
     * 构造函数 / Constructor
     * 
     * @param errorType 错误类型 / Error type
     * @param message 错误消息 / Error message
     * @param cause 原因 / Cause
     */
    public ImageProcessingException(ErrorType errorType, String message, Throwable cause) {
        super(message, cause);
        this.errorType = errorType;
        this.detailMessage = message;
    }
    
    /**
     * 构造函数 / Constructor
     * 
     * @param errorType 错误类型 / Error type
     * @param processorName 处理器名称 / Processor name
     * @param message 错误消息 / Error message
     */
    public ImageProcessingException(ErrorType errorType, String processorName, String message) {
        super(String.format("[%s] %s: %s", errorType, processorName, message));
        this.errorType = errorType;
        this.processorName = processorName;
        this.detailMessage = message;
    }
    
    /**
     * 构造函数 / Constructor
     * 
     * @param errorType 错误类型 / Error type
     * @param processorName 处理器名称 / Processor name
     * @param message 错误消息 / Error message
     * @param cause 原因 / Cause
     */
    public ImageProcessingException(ErrorType errorType, String processorName, String message, Throwable cause) {
        super(String.format("[%s] %s: %s", errorType, processorName, message), cause);
        this.errorType = errorType;
        this.processorName = processorName;
        this.detailMessage = message;
    }
    
    /**
     * 获取错误类型 / Get Error Type
     * 
     * @return 错误类型 / Error type
     */
    public ErrorType getErrorType() {
        return errorType;
    }
    
    /**
     * 获取详细消息 / Get Detail Message
     * 
     * @return 详细消息 / Detail message
     */
    public String getDetailMessage() {
        return detailMessage;
    }
    
    /**
     * 获取处理器名称 / Get Processor Name
     * 
     * @return 处理器名称 / Processor name
     */
    public String getProcessorName() {
        return processorName;
    }
    
    /**
     * 设置处理器名称 / Set Processor Name
     * 
     * @param processorName 处理器名称 / Processor name
     */
    public void setProcessorName(String processorName) {
        this.processorName = processorName;
    }
    
    /**
     * 获取格式化的错误信息 / Get Formatted Error Message
     * 
     * @return 格式化的错误信息 / Formatted error message
     */
    public String getFormattedMessage() {
        StringBuilder sb = new StringBuilder();
        sb.append("ImageProcessingException: ");
        
        if (processorName != null) {
            sb.append("[").append(processorName).append("] ");
        }
        
        sb.append(errorType).append(" - ").append(detailMessage);
        
        if (getCause() != null) {
            sb.append(" (Caused by: ").append(getCause().getMessage()).append(")");
        }
        
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return getFormattedMessage();
    }
    
    // ========== 静态工厂方法 / Static Factory Methods ==========
    
    /**
     * 创建无效输入异常 / Create Invalid Input Exception
     */
    public static ImageProcessingException invalidInput(String message) {
        return new ImageProcessingException(ErrorType.INVALID_INPUT, message);
    }
    
    /**
     * 创建无效参数异常 / Create Invalid Parameters Exception
     */
    public static ImageProcessingException invalidParameters(String message) {
        return new ImageProcessingException(ErrorType.INVALID_PARAMETERS, message);
    }
    
    /**
     * 创建处理失败异常 / Create Processing Failed Exception
     */
    public static ImageProcessingException processingFailed(String processorName, String message) {
        return new ImageProcessingException(ErrorType.PROCESSING_FAILED, processorName, message);
    }
    
    /**
     * 创建处理失败异常 / Create Processing Failed Exception
     */
    public static ImageProcessingException processingFailed(String processorName, String message, Throwable cause) {
        return new ImageProcessingException(ErrorType.PROCESSING_FAILED, processorName, message, cause);
    }
    
    /**
     * 创建内存不足异常 / Create Insufficient Memory Exception
     */
    public static ImageProcessingException insufficientMemory(String message) {
        return new ImageProcessingException(ErrorType.INSUFFICIENT_MEMORY, message);
    }
    
    /**
     * 创建算法错误异常 / Create Algorithm Error Exception
     */
    public static ImageProcessingException algorithmError(String processorName, String message) {
        return new ImageProcessingException(ErrorType.ALGORITHM_ERROR, processorName, message);
    }
    
    /**
     * 创建不支持操作异常 / Create Unsupported Operation Exception
     */
    public static ImageProcessingException unsupportedOperation(String operation) {
        return new ImageProcessingException(ErrorType.UNSUPPORTED_OPERATION, 
                                          "Unsupported operation: " + operation);
    }
    
    /**
     * 创建超时异常 / Create Timeout Exception
     */
    public static ImageProcessingException timeout(String processorName, long timeoutMs) {
        return new ImageProcessingException(ErrorType.TIMEOUT, processorName, 
                                          String.format("Processing timeout after %d ms", timeoutMs));
    }
    
    /**
     * 创建GPU错误异常 / Create GPU Error Exception
     */
    public static ImageProcessingException gpuError(String message) {
        return new ImageProcessingException(ErrorType.GPU_ERROR, message);
    }
    
    /**
     * 创建GPU错误异常 / Create GPU Error Exception
     */
    public static ImageProcessingException gpuError(String message, Throwable cause) {
        return new ImageProcessingException(ErrorType.GPU_ERROR, message, cause);
    }
}