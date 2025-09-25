package com.reremouse.lab.audio.core;

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
    
    /** 序列化版本ID / Serialization version ID */
    private static final long serialVersionUID = 1L;
    
    /** 不支持的格式 / Unsupported format */
    private final String unsupportedFormat;
    
    /** 支持的格式列表 / List of supported formats */
    private final String[] supportedFormats;
    
    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     */
    public UnsupportedAudioFormatException(String message) {
        super(message);
        this.unsupportedFormat = null;
        this.supportedFormats = null;
    }
    
    /**
     * 构造函数 / Constructor
     *
     * @param message 异常消息 / Exception message
     * @param cause 原因异常 / Cause exception
     */
    public UnsupportedAudioFormatException(String message, Throwable cause) {
        super(message, cause);
        this.unsupportedFormat = null;
        this.supportedFormats = null;
    }
    
    /**
     * 构造函数（包含格式信息）/ Constructor (with format information)
     *
     * @param message 异常消息 / Exception message
     * @param unsupportedFormat 不支持的格式 / Unsupported format
     * @param supportedFormats 支持的格式列表 / List of supported formats
     */
    public UnsupportedAudioFormatException(String message, String unsupportedFormat, String[] supportedFormats) {
        super(message);
        this.unsupportedFormat = unsupportedFormat;
        this.supportedFormats = supportedFormats != null ? supportedFormats.clone() : null;
    }
    
    /**
     * 构造函数（包含格式信息和原因）/ Constructor (with format information and cause)
     *
     * @param message 异常消息 / Exception message
     * @param unsupportedFormat 不支持的格式 / Unsupported format
     * @param supportedFormats 支持的格式列表 / List of supported formats
     * @param cause 原因异常 / Cause exception
     */
    public UnsupportedAudioFormatException(String message, String unsupportedFormat, 
                                         String[] supportedFormats, Throwable cause) {
        super(message, cause);
        this.unsupportedFormat = unsupportedFormat;
        this.supportedFormats = supportedFormats != null ? supportedFormats.clone() : null;
    }
    
    /**
     * 获取不支持的格式 / Get unsupported format
     *
     * @return 不支持的格式 / Unsupported format
     */
    public String getUnsupportedFormat() {
        return unsupportedFormat;
    }
    
    /**
     * 获取支持的格式列表 / Get list of supported formats
     *
     * @return 支持的格式列表 / List of supported formats
     */
    public String[] getSupportedFormats() {
        return supportedFormats != null ? supportedFormats.clone() : null;
    }
    
    /**
     * 检查是否有格式信息 / Check if has format information
     *
     * @return 如果有格式信息返回true / True if has format information
     */
    public boolean hasFormatInfo() {
        return unsupportedFormat != null || supportedFormats != null;
    }
    
    /**
     * 获取详细的错误信息 / Get detailed error information
     *
     * @return 详细错误信息 / Detailed error information
     */
    public String getDetailedMessage() {
        StringBuilder sb = new StringBuilder(getMessage());
        
        if (unsupportedFormat != null) {
            sb.append("\n不支持的格式 / Unsupported format: ").append(unsupportedFormat);
        }
        
        if (supportedFormats != null && supportedFormats.length > 0) {
            sb.append("\n支持的格式 / Supported formats: ");
            for (int i = 0; i < supportedFormats.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(supportedFormats[i]);
            }
        }
        
        return sb.toString();
    }
    
    /**
     * 创建格式不支持异常 / Create format not supported exception
     *
     * @param format 不支持的格式 / Unsupported format
     * @param supportedFormats 支持的格式列表 / List of supported formats
     * @return 异常实例 / Exception instance
     */
    public static UnsupportedAudioFormatException formatNotSupported(String format, String... supportedFormats) {
        String message = String.format("音频格式 '%s' 不被支持 / Audio format '%s' is not supported", format, format);
        return new UnsupportedAudioFormatException(message, format, supportedFormats);
    }
    
    /**
     * 创建编码不支持异常 / Create encoding not supported exception
     *
     * @param format 格式 / Format
     * @param operation 操作类型 / Operation type
     * @return 异常实例 / Exception instance
     */
    public static UnsupportedAudioFormatException encodingNotSupported(String format, String operation) {
        String message = String.format("格式 '%s' 的 %s 操作不被支持 / %s operation for format '%s' is not supported", 
                                      format, operation, operation, format);
        return new UnsupportedAudioFormatException(message, format, null);
    }
    
    /**
     * 创建解码不支持异常 / Create decoding not supported exception
     *
     * @param format 格式 / Format
     * @return 异常实例 / Exception instance
     */
    public static UnsupportedAudioFormatException decodingNotSupported(String format) {
        return encodingNotSupported(format, "解码 / Decoding");
    }
    
    /**
     * 创建编码不支持异常 / Create encoding not supported exception
     *
     * @param format 格式 / Format
     * @return 异常实例 / Exception instance
     */
    public static UnsupportedAudioFormatException encodingNotSupported(String format) {
        return encodingNotSupported(format, "编码 / Encoding");
    }
    
    /**
     * 创建参数不支持异常 / Create parameter not supported exception
     *
     * @param parameter 参数名 / Parameter name
     * @param value 参数值 / Parameter value
     * @param format 格式 / Format
     * @return 异常实例 / Exception instance
     */
    public static UnsupportedAudioFormatException parameterNotSupported(String parameter, Object value, String format) {
        String message = String.format("格式 '%s' 不支持参数 '%s' 的值 '%s' / Format '%s' does not support parameter '%s' with value '%s'", 
                                      format, parameter, value, format, parameter, value);
        return new UnsupportedAudioFormatException(message, format, null);
    }
    
    @Override
    public String toString() {
        if (hasFormatInfo()) {
            return getClass().getSimpleName() + ": " + getDetailedMessage();
        } else {
            return super.toString();
        }
    }
}