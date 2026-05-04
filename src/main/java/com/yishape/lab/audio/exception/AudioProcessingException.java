package com.yishape.lab.audio.exception;

/**
 * 音频处理异常类 / Audio Processing Exception Class
 * <p>
 * 当音频处理过程中发生错误时抛出此异常。
 * Exception thrown when errors occur during audio processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioProcessingException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * 默认构造函数 / Default constructor
     */
    public AudioProcessingException() {
        super();
    }

    /**
     * 带错误消息的构造函数 / Constructor with error message
     *
     * @param message 错误消息 / Error message
     */
    public AudioProcessingException(String message) {
        super(message);
    }

    /**
     * 带错误消息和原因的构造函数 / Constructor with error message and cause
     *
     * @param message 错误消息 / Error message
     * @param cause 原因 / Cause
     */
    public AudioProcessingException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * 带原因的构造函数 / Constructor with cause
     *
     * @param cause 原因 / Cause
     */
    public AudioProcessingException(Throwable cause) {
        super(cause);
    }
}