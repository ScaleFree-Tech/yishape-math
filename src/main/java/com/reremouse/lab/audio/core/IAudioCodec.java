package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;

/**
 * 音频编解码器接口 / Audio Codec Interface
 * <p>
 * 定义音频编解码器的基本操作，包括音频格式转换、压缩、解压缩等。
 * 所有音频编解码器都应该实现此接口，确保一致的API设计。
 * </p>
 * <p>
 * Defines basic operations for audio codecs, including audio format conversion, compression, decompression, etc.
 * All audio codecs should implement this interface to ensure consistent API design.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioCodec {
    
    /**
     * 音频编解码器类型枚举 / Audio Codec Type Enum
     */
    enum CodecType {
        PCM("PCM", "Pulse Code Modulation"),
        MP3("MP3", "MPEG-1 Audio Layer III"),
        AAC("AAC", "Advanced Audio Coding"),
        FLAC("FLAC", "Free Lossless Audio Codec"),
        OGG("OGG", "Ogg Vorbis"),
        WAV("WAV", "Waveform Audio File Format"),
        AIFF("AIFF", "Audio Interchange File Format"),
        WMA("WMA", "Windows Media Audio"),
        M4A("M4A", "MPEG-4 Audio"),
        OPUS("OPUS", "Opus Audio Codec");
        
        private final String extension;
        private final String fullName;
        
        CodecType(String extension, String fullName) {
            this.extension = extension;
            this.fullName = fullName;
        }
        
        public String getExtension() { return extension; }
        public String getFullName() { return fullName; }
        
        @Override
        public String toString() {
            return extension + " (" + fullName + ")";
        }
    }
    
    /**
     * 编解码质量枚举 / Codec Quality Enum
     */
    enum Quality {
        LOW("低质量", "Low Quality", 64),
        MEDIUM("中等质量", "Medium Quality", 128),
        HIGH("高质量", "High Quality", 256),
        VERY_HIGH("超高质量", "Very High Quality", 320),
        LOSSLESS("无损", "Lossless", -1);
        
        private final String chineseName;
        private final String englishName;
        private final int bitrate; // -1 for lossless
        
        Quality(String chineseName, String englishName, int bitrate) {
            this.chineseName = chineseName;
            this.englishName = englishName;
            this.bitrate = bitrate;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
        public int getBitrate() { return bitrate; }
        
        @Override
        public String toString() {
            return chineseName + " / " + englishName + 
                   (bitrate > 0 ? " (" + bitrate + " kbps)" : "");
        }
    }
    
    /**
     * 编码音频数据 / Encode audio data
     * <p>
     * 将音频数据编码为指定格式。
     * Encode audio data to specified format.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param codecType 目标编解码器类型 / Target codec type
     * @param quality 编码质量 / Encoding quality
     * @return 编码后的字节数据 / Encoded byte data
     * @throws AudioProcessingException 当编码过程中发生错误时抛出 / Thrown when error occurs during encoding
     */
    byte[] encode(AudioData audioData, CodecType codecType, Quality quality) throws AudioProcessingException;
    
    /**
     * 解码音频数据 / Decode audio data
     * <p>
     * 将编码的字节数据解码为音频数据。
     * Decode encoded byte data to audio data.
     * </p>
     *
     * @param encodedData 编码的字节数据 / Encoded byte data
     * @param codecType 源编解码器类型 / Source codec type
     * @return 解码后的音频数据 / Decoded audio data
     * @throws AudioProcessingException 当解码过程中发生错误时抛出 / Thrown when error occurs during decoding
     */
    AudioData decode(byte[] encodedData, CodecType codecType) throws AudioProcessingException;
    
    /**
     * 转换音频格式 / Convert audio format
     * <p>
     * 直接转换音频数据的格式。
     * Directly convert audio data format.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param targetCodecType 目标编解码器类型 / Target codec type
     * @param quality 转换质量 / Conversion quality
     * @return 转换后的音频数据 / Converted audio data
     * @throws AudioProcessingException 当转换过程中发生错误时抛出 / Thrown when error occurs during conversion
     */
    AudioData convert(AudioData audioData, CodecType targetCodecType, Quality quality) throws AudioProcessingException;
    
    /**
     * 检查是否支持编码 / Check if supports encoding
     * <p>
     * 检查编解码器是否支持编码到指定格式。
     * Check if codec supports encoding to specified format.
     * </p>
     *
     * @param codecType 编解码器类型 / Codec type
     * @return 如果支持编码返回true / Return true if encoding is supported
     */
    boolean supportsEncoding(CodecType codecType);
    
    /**
     * 检查是否支持解码 / Check if supports decoding
     * <p>
     * 检查编解码器是否支持从指定格式解码。
     * Check if codec supports decoding from specified format.
     * </p>
     *
     * @param codecType 编解码器类型 / Codec type
     * @return 如果支持解码返回true / Return true if decoding is supported
     */
    boolean supportsDecoding(CodecType codecType);
    
    /**
     * 获取支持的编码格式 / Get supported encoding formats
     * <p>
     * 返回编解码器支持的所有编码格式。
     * Return all encoding formats supported by the codec.
     * </p>
     *
     * @return 支持的编解码器类型数组 / Array of supported codec types
     */
    CodecType[] getSupportedEncodingFormats();
    
    /**
     * 获取支持的解码格式 / Get supported decoding formats
     * <p>
     * 返回编解码器支持的所有解码格式。
     * Return all decoding formats supported by the codec.
     * </p>
     *
     * @return 支持的编解码器类型数组 / Array of supported codec types
     */
    CodecType[] getSupportedDecodingFormats();
    
    /**
     * 设置编码参数 / Set encoding parameters
     * <p>
     * 设置编码过程的参数。
     * Set parameters for encoding process.
     * </p>
     *
     * @param key 参数键 / Parameter key
     * @param value 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数无效时抛出 / Thrown when parameter is invalid
     */
    void setEncodingParameter(String key, Object value) throws IllegalArgumentException;
    
    /**
     * 获取编码参数 / Get encoding parameter
     * <p>
     * 获取指定编码参数的值。
     * Get value of specified encoding parameter.
     * </p>
     *
     * @param key 参数键 / Parameter key
     * @return 参数值 / Parameter value
     * @throws IllegalArgumentException 当参数键不存在时抛出 / Thrown when parameter key doesn't exist
     */
    Object getEncodingParameter(String key) throws IllegalArgumentException;
    
    /**
     * 获取编解码器信息 / Get codec information
     * <p>
     * 返回编解码器的详细信息。
     * Return detailed information about the codec.
     * </p>
     *
     * @return 编解码器信息字符串 / Codec information string
     */
    String getCodecInfo();
    
    /**
     * 估算编码后的文件大小 / Estimate encoded file size
     * <p>
     * 根据音频数据和编码参数估算编码后的文件大小。
     * Estimate encoded file size based on audio data and encoding parameters.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param codecType 编解码器类型 / Codec type
     * @param quality 编码质量 / Encoding quality
     * @return 估算的文件大小（字节） / Estimated file size (bytes)
     */
    long estimateEncodedSize(AudioData audioData, CodecType codecType, Quality quality);
}