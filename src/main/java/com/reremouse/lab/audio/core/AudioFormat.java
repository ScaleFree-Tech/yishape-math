package com.reremouse.lab.audio.core;

/**
 * Audio Format Enum
 * <p>
 * Defines supported audio format types with their characteristics and capabilities.
 * Provides methods for format detection, validation, and classification.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public enum AudioFormat {
    
    /** WAV format - Uncompressed audio format */
    WAV("wav", "Waveform Audio File Format", true, false, "audio/wav"),
    
    /** MP3 format - Compressed audio format */
    MP3("mp3", "MPEG Audio Layer III", false, true, "audio/mpeg"),
    
    /** FLAC format - Lossless compressed audio format */
    FLAC("flac", "Free Lossless Audio Codec", true, false, "audio/flac"),
    
    /** OGG format - Open source compressed audio format */
    OGG("ogg", "Ogg Vorbis", false, true, "audio/ogg"),
    
    /** M4A format - MPEG-4 audio format */
    M4A("m4a", "MPEG-4 Audio", false, true, "audio/mp4"),
    
    /** AAC format - Advanced Audio Coding */
    AAC("aac", "Advanced Audio Coding", false, true, "audio/aac"),
    
    /** AIFF format - Audio Interchange File Format */
    AIFF("aiff", "Audio Interchange File Format", true, false, "audio/aiff"),
    
    /** AU format - Audio File Format */
    AU("au", "Audio File Format", true, false, "audio/basic"),
    
    /** RAW format - Raw audio data */
    RAW("raw", "Raw Audio Data", true, false, "audio/raw"),
    
    /** WMA format - Windows Media Audio */
    WMA("wma", "Windows Media Audio", false, true, "audio/x-ms-wma"),
    
    /** OPUS format - Modern audio codec */
    OPUS("opus", "Opus Audio Codec", false, true, "audio/opus"),
    
    /** APE format - Monkey's Audio */
    APE("ape", "Monkey's Audio", true, false, "audio/ape"),
    
    /** DSD format - Direct Stream Digital */
    DSD("dsd", "Direct Stream Digital", true, false, "audio/dsd"),
    
    /** PCM format - Pulse Code Modulation */
    PCM("pcm", "Pulse Code Modulation", true, false, "audio/pcm");
    
    private final String extension;
    private final String description;
    private final boolean lossless;
    private final boolean compressed;
    private final String mimeType;
    
    /**
     * Constructor
     *
     * @param extension File extension
     * @param description Format description
     * @param lossless Whether the format is lossless
     * @param compressed Whether the format uses compression
     * @param mimeType MIME type for the format
     */
    AudioFormat(String extension, String description, boolean lossless, boolean compressed, String mimeType) {
        this.extension = extension;
        this.description = description;
        this.lossless = lossless;
        this.compressed = compressed;
        this.mimeType = mimeType;
    }
    
    /**
     * Get file extension
     *
     * @return File extension
     */
    public String getExtension() {
        return extension;
    }
    
    /**
     * Get format description
     *
     * @return Format description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Get MIME type
     *
     * @return MIME type
     */
    public String getMimeType() {
        return mimeType;
    }
    
    /**
     * Check if lossless format
     *
     * @return True if lossless format
     */
    public boolean isLossless() {
        return lossless;
    }
    
    /**
     * Check if lossy format
     *
     * @return True if lossy format
     */
    public boolean isLossy() {
        return !lossless;
    }
    
    /**
     * Check if compressed format
     *
     * @return True if compressed format
     */
    public boolean isCompressed() {
        return compressed;
    }
    
    /**
     * Check if uncompressed format
     *
     * @return True if uncompressed format
     */
    public boolean isUncompressed() {
        return !compressed;
    }
    
    /**
     * Get audio format by file extension
     *
     * @param extension File extension
     * @return Corresponding audio format, null if not found
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
     * Get audio format by MIME type
     *
     * @param mimeType MIME type
     * @return Corresponding audio format, null if not found
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
     * Get audio format by filename
     *
     * @param filename Filename with extension
     * @return Corresponding audio format, null if not found
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
     * Check if format supports metadata
     *
     * @return True if format supports metadata
     */
    public boolean supportsMetadata() {
        return this != RAW && this != PCM && this != AU;
    }
    
    /**
     * Check if format supports streaming
     *
     * @return True if format supports streaming
     */
    public boolean supportsStreaming() {
        return this == MP3 || this == OGG || this == AAC || this == OPUS;
    }
    
    /**
     * Check if format is suitable for high-quality audio
     *
     * @return True if suitable for high-quality audio
     */
    public boolean isHighQuality() {
        return isLossless() || this == OPUS;
    }
    
    /**
     * Check if format is suitable for web streaming
     *
     * @return True if suitable for web streaming
     */
    public boolean isWebCompatible() {
        return this == MP3 || this == OGG || this == AAC || this == OPUS || this == WAV;
    }
    
    /**
     * Get typical compression ratio compared to WAV
     *
     * @return Compression ratio (1.0 = no compression, 0.1 = 10x compression)
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
     * Get maximum supported sample rate for this format
     *
     * @return Maximum sample rate in Hz
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
     * Get maximum supported bit depth for this format
     *
     * @return Maximum bit depth
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
     * Check if format supports multiple channels
     *
     * @return True if supports multiple channels
     */
    public boolean supportsMultiChannel() {
        return this != AU; // AU typically supports only mono/stereo
    }
    
    /**
     * Get all lossless formats
     *
     * @return Array of lossless formats
     */
    public static AudioFormat[] getLosslessFormats() {
        return new AudioFormat[]{WAV, FLAC, AIFF, AU, RAW, APE, DSD, PCM};
    }
    
    /**
     * Get all lossy formats
     *
     * @return Array of lossy formats
     */
    public static AudioFormat[] getLossyFormats() {
        return new AudioFormat[]{MP3, OGG, M4A, AAC, WMA, OPUS};
    }
    
    /**
     * Get all web-compatible formats
     *
     * @return Array of web-compatible formats
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
     * Get detailed format information
     *
     * @return Detailed format information string
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