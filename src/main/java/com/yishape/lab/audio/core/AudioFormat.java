package com.yishape.lab.audio.core;

/**
 * Audio Format Enum
 * <p>
 * 定义支持的音频格式类型及其特性和功能。提供格式检测、验证和分类的方法。
 * 支持WAV、MP3、FLAC、OGG、M4A、AAC、AIFF、AU、RAW、WMA、OPUS、APE、DSD、PCM等格式。
 * </p>
 * <p>
 * Defines supported audio format types with their characteristics and capabilities.
 * Provides methods for format detection, validation, and classification.
 * Supports WAV, MP3, FLAC, OGG, M4A, AAC, AIFF, AU, RAW, WMA, OPUS, APE, DSD, PCM formats.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public enum AudioFormat {
    
    /** WAV格式 - 未压缩音频格式 / WAV format - Uncompressed audio format */
    WAV("wav", "Waveform Audio File Format", true, false, "audio/wav"),

    /** MP3格式 - 压缩音频格式 / MP3 format - Compressed audio format */
    MP3("mp3", "MPEG Audio Layer III", false, true, "audio/mpeg"),

    /** FLAC格式 - 无损压缩音频格式 / FLAC format - Lossless compressed audio format */
    FLAC("flac", "Free Lossless Audio Codec", true, false, "audio/flac"),

    /** OGG格式 - 开源压缩音频格式 / OGG format - Open source compressed audio format */
    OGG("ogg", "Ogg Vorbis", false, true, "audio/ogg"),

    /** M4A格式 - MPEG-4音频格式 / M4A format - MPEG-4 audio format */
    M4A("m4a", "MPEG-4 Audio", false, true, "audio/mp4"),

    /** AAC格式 - 高级音频编码 / AAC format - Advanced Audio Coding */
    AAC("aac", "Advanced Audio Coding", false, true, "audio/aac"),

    /** AIFF格式 - 音频交换文件格式 / AIFF format - Audio Interchange File Format */
    AIFF("aiff", "Audio Interchange File Format", true, false, "audio/aiff"),

    /** AU格式 - 音频文件格式 / AU format - Audio File Format */
    AU("au", "Audio File Format", true, false, "audio/basic"),

    /** RAW格式 - 原始音频数据 / RAW format - Raw audio data */
    RAW("raw", "Raw Audio Data", true, false, "audio/raw"),

    /** WMA格式 - Windows Media Audio / WMA format - Windows Media Audio */
    WMA("wma", "Windows Media Audio", false, true, "audio/x-ms-wma"),

    /** OPUS格式 - 现代音频编解码器 / OPUS format - Modern audio codec */
    OPUS("opus", "Opus Audio Codec", false, true, "audio/opus"),

    /** APE格式 - 猴子音频 / APE format - Monkey's Audio */
    APE("ape", "Monkey's Audio", true, false, "audio/ape"),

    /** DSD格式 - 直接流数字 / DSD format - Direct Stream Digital */
    DSD("dsd", "Direct Stream Digital", true, false, "audio/dsd"),

    /** PCM格式 - 脉冲编码调制 / PCM format - Pulse Code Modulation */
    PCM("pcm", "Pulse Code Modulation", true, false, "audio/pcm");
    
    private final String extension;
    private final String description;
    private final boolean lossless;
    private final boolean compressed;
    private final String mimeType;
    
    /**
     * 构造函数 / Constructor
     *
     * @param extension 文件扩展名 / File extension
     * @param description 格式描述 / Format description
     * @param lossless 是否为无损格式 / Whether the format is lossless
     * @param compressed 是否使用压缩 / Whether the format uses compression
     * @param mimeType 格式的MIME类型 / MIME type for the format
     */
    AudioFormat(String extension, String description, boolean lossless, boolean compressed, String mimeType) {
        this.extension = extension;
        this.description = description;
        this.lossless = lossless;
        this.compressed = compressed;
        this.mimeType = mimeType;
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
     * 获取MIME类型 / Get MIME type
     *
     * @return MIME类型 / MIME type
     */
    public String getMimeType() {
        return mimeType;
    }
    
    /**
     * 检查是否为无损格式 / Check if lossless format
     *
     * @return 如果是无损格式则返回true / True if lossless format
     */
    public boolean isLossless() {
        return lossless;
    }
    
    /**
     * 检查是否为有损格式 / Check if lossy format
     *
     * @return 如果是有损格式则返回true / True if lossy format
     */
    public boolean isLossy() {
        return !lossless;
    }
    
    /**
     * 检查是否为压缩格式 / Check if compressed format
     *
     * @return 如果是压缩格式则返回true / True if compressed format
     */
    public boolean isCompressed() {
        return compressed;
    }
    
    /**
     * 检查是否为非压缩格式 / Check if uncompressed format
     *
     * @return 如果是非压缩格式则返回true / True if uncompressed format
     */
    public boolean isUncompressed() {
        return !compressed;
    }
    
    /**
     * 根据文件扩展名获取音频格式 / Get audio format by file extension
     *
     * @param extension 文件扩展名 / File extension
     * @return 对应的音频格式，如果未找到则返回null / Corresponding audio format, null if not found
     */
    public static AudioFormat fromExtension(String extension) {
        if (extension == null || extension.trim().isEmpty()) {
            return null;
        }
        
        String ext = extension.toLowerCase().replace(".", "").trim();
        for (AudioFormat format : values()) {
            if (format.extension.equals(ext)) {
                return format;
            }
        }
        return null;
    }
    
    /**
     * 根据MIME类型获取音频格式 / Get audio format by MIME type
     *
     * @param mimeType MIME类型 / MIME type
     * @return 对应的音频格式，如果未找到则返回null / Corresponding audio format, null if not found
     */
    public static AudioFormat fromMimeType(String mimeType) {
        if (mimeType == null || mimeType.trim().isEmpty()) {
            return null;
        }
        
        String mime = mimeType.toLowerCase().trim();
        for (AudioFormat format : values()) {
            if (format.mimeType.equals(mime)) {
                return format;
            }
        }
        return null;
    }
    
    /**
     * 根据文件名获取音频格式 / Get audio format by filename
     *
     * @param filename 带扩展名的文件名 / Filename with extension
     * @return 对应的音频格式，如果未找到则返回null / Corresponding audio format, null if not found
     */
    public static AudioFormat fromFilename(String filename) {
        if (filename == null || filename.trim().isEmpty()) {
            return null;
        }
        
        int lastDotIndex = filename.lastIndexOf('.');
        if (lastDotIndex == -1 || lastDotIndex == filename.length() - 1) {
            return null;
        }
        
        String extension = filename.substring(lastDotIndex + 1);
        return fromExtension(extension);
    }
    
    /**
     * 检查格式是否支持元数据 / Check if format supports metadata
     *
     * @return 如果支持元数据则返回true / True if format supports metadata
     */
    public boolean supportsMetadata() {
        return this != RAW && this != PCM && this != AU;
    }
    
    /**
     * 检查格式是否支持流媒体 / Check if format supports streaming
     *
     * @return 如果支持流媒体则返回true / True if format supports streaming
     */
    public boolean supportsStreaming() {
        return this == MP3 || this == OGG || this == AAC || this == OPUS;
    }
    
    /**
     * 检查格式是否适合高质量音频 / Check if format is suitable for high-quality audio
     *
     * @return 如果适合高质量音频则返回true / True if suitable for high-quality audio
     */
    public boolean isHighQuality() {
        return isLossless() || this == OPUS;
    }
    
    /**
     * 检查格式是否适合网络流媒体 / Check if format is suitable for web streaming
     *
     * @return 如果适合网络流媒体则返回true / True if suitable for web streaming
     */
    public boolean isWebCompatible() {
        return this == MP3 || this == OGG || this == AAC || this == OPUS || this == WAV;
    }
    
    /**
     * 获取与WAV相比的典型压缩比 / Get typical compression ratio compared to WAV
     *
     * @return 压缩比（1.0表示无压缩，0.1表示10倍压缩）/ Compression ratio (1.0 = no compression, 0.1 = 10x compression)
     */
    public double getTypicalCompressionRatio() {
        switch (this) {
            case WAV:
            case AIFF:
            case AU:
            case RAW:
            case PCM:
                return 1.0; // No compression
            case FLAC:
            case APE:
                return 0.6; // ~40% compression
            case MP3:
            case AAC:
            case WMA:
                return 0.1; // ~90% compression
            case OGG:
            case OPUS:
                return 0.12; // ~88% compression
            case M4A:
                return 0.15; // ~85% compression
            case DSD:
                return 0.5; // ~50% compression
            default:
                return 1.0;
        }
    }
    
    /**
     * 获取此格式支持的最大采样率 / Get maximum supported sample rate for this format
     *
     * @return 最大采样率（Hz）/ Maximum sample rate in Hz
     */
    public int getMaxSampleRate() {
        switch (this) {
            case DSD:
                return 2822400; // DSD64
            case FLAC:
            case WAV:
            case AIFF:
                return 192000; // High-resolution audio
            case PCM:
            case RAW:
                return 384000; // Professional audio
            case APE:
                return 192000;
            case MP3:
                return 48000;
            case AAC:
            case M4A:
                return 96000;
            case OGG:
                return 192000;
            case OPUS:
                return 48000;
            case WMA:
                return 48000;
            case AU:
                return 44100;
            default:
                return 44100;
        }
    }
    
    /**
     * 获取此格式支持的最大位深度 / Get maximum supported bit depth for this format
     *
     * @return 最大位深度 / Maximum bit depth
     */
    public int getMaxBitDepth() {
        switch (this) {
            case WAV:
            case FLAC:
            case AIFF:
            case PCM:
            case RAW:
                return 32; // 32-bit float or integer
            case APE:
                return 24;
            case DSD:
                return 1; // 1-bit
            case AU:
                return 16;
            default:
                return 16; // Compressed formats typically use 16-bit internally
        }
    }
    
    /**
     * 检查格式是否支持多通道 / Check if format supports multiple channels
     *
     * @return 如果支持多通道则返回true / True if supports multiple channels
     */
    public boolean supportsMultiChannel() {
        return this != AU; // AU typically supports only mono/stereo
    }
    
    /**
     * 获取所有无损格式 / Get all lossless formats
     *
     * @return 无损格式数组 / Array of lossless formats
     */
    public static AudioFormat[] getLosslessFormats() {
        return new AudioFormat[]{WAV, FLAC, AIFF, AU, RAW, APE, DSD, PCM};
    }
    
    /**
     * 获取所有有损格式 / Get all lossy formats
     *
     * @return 有损格式数组 / Array of lossy formats
     */
    public static AudioFormat[] getLossyFormats() {
        return new AudioFormat[]{MP3, OGG, M4A, AAC, WMA, OPUS};
    }
    
    /**
     * 获取所有网络兼容格式 / Get all web-compatible formats
     *
     * @return 网络兼容格式数组 / Array of web-compatible formats
     */
    public static AudioFormat[] getWebCompatibleFormats() {
        return new AudioFormat[]{MP3, OGG, AAC, OPUS, WAV};
    }
    
    @Override
    public String toString() {
        return String.format("%s (%s) - %s", 
                description, 
                extension.toUpperCase(), 
                lossless ? "Lossless" : "Lossy");
    }
    
    /**
     * 获取详细的格式信息 / Get detailed format information
     *
     * @return 详细的格式信息字符串 / Detailed format information string
     */
    public String getDetailedInfo() {
        return String.format("%s\n" +
                "Extension: .%s\n" +
                "MIME Type: %s\n" +
                "Quality: %s\n" +
                "Compression: %s\n" +
                "Max Sample Rate: %d Hz\n" +
                "Max Bit Depth: %d bits\n" +
                "Metadata Support: %s\n" +
                "Streaming Support: %s\n" +
                "Web Compatible: %s",
                description,
                extension,
                mimeType,
                lossless ? "Lossless" : "Lossy",
                compressed ? "Yes" : "No",
                getMaxSampleRate(),
                getMaxBitDepth(),
                supportsMetadata() ? "Yes" : "No",
                supportsStreaming() ? "Yes" : "No",
                isWebCompatible() ? "Yes" : "No");
    }
}