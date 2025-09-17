package com.reremouse.lab.audio;

/**
 * 不支持的音频格式异常 / Unsupported Audio Format Exception
 * <p>
 * 当尝试处理不支持的音频格式时抛出此异常。
 * Thrown when attempting to process unsupported audio formats.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class UnsupportedAudioFormatException extends Exception {
    
    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     */
    public UnsupportedAudioFormatException(String message) {
        super(message);
    }
    
    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     * @param cause 原因异常 / Cause exception
     */
    public UnsupportedAudioFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
