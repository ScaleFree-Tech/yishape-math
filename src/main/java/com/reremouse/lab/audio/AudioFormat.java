package com.reremouse.lab.audio;

/**
 * 音频格式枚举 / Audio Format Enum
 * <p>
 * 定义支持的音频格式类型。
 * Defines supported audio format types.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public enum AudioFormat {
    
    /** WAV格式 / WAV format */
    WAV("wav", "Waveform Audio File Format"),
    
    /** MP3格式 / MP3 format */
    MP3("mp3", "MPEG Audio Layer III"),
    
    /** FLAC格式 / FLAC format */
    FLAC("flac", "Free Lossless Audio Codec"),
    
    /** OGG格式 / OGG format */
    OGG("ogg", "Ogg Vorbis"),
    
    /** M4A格式 / M4A format */
    M4A("m4a", "MPEG-4 Audio"),
    
    /** AAC格式 / AAC format */
    AAC("aac", "Advanced Audio Coding"),
    
    /** AIFF格式 / AIFF format */
    AIFF("aiff", "Audio Interchange File Format"),
    
    /** AU格式 / AU format */
    AU("au", "Audio File Format"),
    
    /** RAW格式 / RAW format */
    RAW("raw", "Raw Audio Data");
    
    private final String extension;
    private final String description;
    
    /**
     * 构造函数 / Constructor
     *
     * @param extension 文件扩展名 / File extension
     * @param description 格式描述 / Format description
     */
    AudioFormat(String extension, String description) {
        this.extension = extension;
        this.description = description;
    }
    
    /**
     * 获取文件扩展名 / Get file extension
     *
     * @return 文件扩展名 / File extension
     */
    public String getExtension() {
        return extension;
    }
    
    /**
     * 获取格式描述 / Get format description
     *
     * @return 格式描述 / Format description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * 根据文件扩展名获取音频格式 / Get audio format by file extension
     *
     * @param extension 文件扩展名 / File extension
     * @return 对应的音频格式，如果未找到返回null / Corresponding audio format, null if not found
     */
    public static AudioFormat fromExtension(String extension) {
        if (extension == null) {
            return null;
        }
        
        String ext = extension.toLowerCase().replace(".", "");
        for (AudioFormat format : values()) {
            if (format.extension.equals(ext)) {
                return format;
            }
        }
        return null;
    }
    
    /**
     * 检查是否为无损格式 / Check if lossless format
     *
     * @return 如果是无损格式返回true / True if lossless format
     */
    public boolean isLossless() {
        return this == WAV || this == FLAC || this == AIFF || this == AU || this == RAW;
    }
    
    /**
     * 检查是否为有损格式 / Check if lossy format
     *
     * @return 如果是有损格式返回true / True if lossy format
     */
    public boolean isLossy() {
        return this == MP3 || this == OGG || this == M4A || this == AAC;
    }
    
    @Override
    public String toString() {
        return String.format("%s (%s)", description, extension.toUpperCase());
    }
}
