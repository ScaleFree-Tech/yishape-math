package com.yishape.lab.audio.exception;

/**
 * 音频分析异常 / Audio Analysis Exception
 * <p>
 * 当音频分析过程中发生错误时抛出此异常。
 * Thrown when an error occurs during audio analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioAnalysisException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * 构造函数 / Constructor
     */
    public AudioAnalysisException() {
        super();
    }

    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     */
    public AudioAnalysisException(String message) {
        super(message);
    }

    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     * @param cause 异常原因 / Exception cause
     */
    public AudioAnalysisException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * 构造函数 / Constructor
     *
     * @param cause 异常原因 / Exception cause
     */
    public AudioAnalysisException(Throwable cause) {
        super(cause);
    }
}