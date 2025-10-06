package com.yishape.lab.audio.core;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javazoom.jl.decoder.Bitstream;
import javazoom.jl.decoder.Header;
import javazoom.jl.decoder.Decoder;
import javazoom.jl.decoder.SampleBuffer;
import javazoom.jl.decoder.JavaLayerException;
import java.io.ByteArrayInputStream;
import java.io.PushbackInputStream;

/**
 * 音频输入输出类 / Audio Input/Output Class
 * <p>
 * 提供音频文件的读取和写入功能，支持多种音频格式。
 * 使用IVector接口存储音频数据，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides audio file reading and writing functionality, supporting multiple audio formats.
 * Uses IVector interface to store audio data, ensuring compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioIO {
    
    /** 默认采样率 / Default sample rate */
    public static final double DEFAULT_SAMPLE_RATE = 44100.0;
    
    /** 默认声道数 / Default number of channels */
    public static final int DEFAULT_CHANNELS = 1;
    
    /** 默认位深度 / Default bit depth */
    public static final int DEFAULT_BIT_DEPTH = 16;
    
    /**
     * 从文件读取音频数据 / Read audio data from file
     * <p>
     * 根据文件扩展名自动识别音频格式并读取。
     * Automatically identifies audio format by file extension and reads.
     * </p>
     *
     * @param filePath 音频文件路径 / Audio file path
     * @return 音频数据对象 / Audio data object
     * @throws IOException 如果文件读取失败 / If file reading fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static AudioData readAudio(String filePath) throws IOException, UnsupportedAudioFormatException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Audio file not found: " + filePath);
        }
        
        String extension = getFileExtension(filePath);
        AudioFormat format = AudioFormat.fromExtension(extension);
        
        if (format == null) {
            throw new UnsupportedAudioFormatException("Unsupported audio format: " + extension);
        }
        
        byte[] fileData = Files.readAllBytes(path);
        return parseAudioData(fileData, format);
    }
    
    /**
     * 从文件读取音频数据（指定格式） / Read audio data from file (specified format)
     *
     * @param filePath 音频文件路径 / Audio file path
     * @param format 音频格式 / Audio format
     * @return 音频数据对象 / Audio data object
     * @throws IOException 如果文件读取失败 / If file reading fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static AudioData readAudio(String filePath, AudioFormat format) throws IOException, UnsupportedAudioFormatException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Audio file not found: " + filePath);
        }
        
        byte[] fileData = Files.readAllBytes(path);
        return parseAudioData(fileData, format);
    }
    
    /**
     * 将音频数据写入文件 / Write audio data to file
     *
     * @param audioData 音频数据 / Audio data
     * @param filePath 输出文件路径 / Output file path
     * @throws IOException 如果文件写入失败 / If file writing fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static void writeAudio(AudioData audioData, String filePath) throws IOException, UnsupportedAudioFormatException {
        String extension = getFileExtension(filePath);
        AudioFormat format = AudioFormat.fromExtension(extension);
        
        if (format == null) {
            throw new UnsupportedAudioFormatException("Unsupported audio format: " + extension);
        }
        
        byte[] audioBytes = encodeAudioData(audioData, format);
        
        try (FileOutputStream fos = new FileOutputStream(filePath)) {
            fos.write(audioBytes);
        }
    }
    
    /**
     * 将音频数据写入文件（指定格式） / Write audio data to file (specified format)
     *
     * @param audioData 音频数据 / Audio data
     * @param filePath 输出文件路径 / Output file path
     * @param format 音频格式 / Audio format
     * @throws IOException 如果文件写入失败 / If file writing fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static void writeAudio(AudioData audioData, String filePath, AudioFormat format) throws IOException, UnsupportedAudioFormatException {
        byte[] audioBytes = encodeAudioData(audioData, format);
        
        try (FileOutputStream fos = new FileOutputStream(filePath)) {
            fos.write(audioBytes);
        }
    }
    
    /**
     * 解析音频数据 / Parse audio data
     *
     * @param fileData 文件字节数据 / File byte data
     * @param format 音频格式 / Audio format
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    private static AudioData parseAudioData(byte[] fileData, AudioFormat format) throws UnsupportedAudioFormatException {
        switch (format) {
            case WAV:
                return parseWAV(fileData);
            case MP3:
                return parseMP3(fileData);
            case FLAC:
                return parseFLAC(fileData);
            case OGG:
                return parseOGG(fileData);
            case M4A:
                return parseM4A(fileData);
            case AAC:
                return parseAAC(fileData);
            case AIFF:
                return parseAIFF(fileData);
            case AU:
                return parseAU(fileData);
            case RAW:
                return parseRAW(fileData);
            default:
                throw new UnsupportedAudioFormatException("Format not yet implemented: " + format);
        }
    }
    
    /**
     * 编码音频数据 / Encode audio data
     *
     * @param audioData 音频数据 / Audio data
     * @param format 音频格式 / Audio format
     * @return 编码后的字节数据 / Encoded byte data
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    private static byte[] encodeAudioData(AudioData audioData, AudioFormat format) throws UnsupportedAudioFormatException {
        switch (format) {
            case WAV:
                return encodeWAV(audioData);
            case MP3:
                return encodeMP3(audioData);
            case FLAC:
                return encodeFLAC(audioData);
            case OGG:
                return encodeOGG(audioData);
            case M4A:
                return encodeM4A(audioData);
            case AAC:
                return encodeAAC(audioData);
            case AIFF:
                return encodeAIFF(audioData);
            case AU:
                return encodeAU(audioData);
            case RAW:
                return encodeRAW(audioData);
            default:
                throw new UnsupportedAudioFormatException("Format not yet implemented: " + format);
        }
    }
    
    /**
     * 解析WAV文件 / Parse WAV file
     *
     * @param fileData WAV文件数据 / WAV file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果WAV格式不支持 / If WAV format is not supported
     */
    private static AudioData parseWAV(byte[] fileData) throws UnsupportedAudioFormatException {
        ByteBuffer buffer = ByteBuffer.wrap(fileData);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // 检查RIFF头 / Check RIFF header
        byte[] riff = new byte[4];
        buffer.get(riff);
        if (!new String(riff).equals("RIFF")) {
            throw new UnsupportedAudioFormatException("Invalid WAV file: missing RIFF header");
        }
        
        // 跳过文件大小 / Skip file size
        buffer.getInt();
        
        // 检查WAVE头 / Check WAVE header
        byte[] wave = new byte[4];
        buffer.get(wave);
        if (!new String(wave).equals("WAVE")) {
            throw new UnsupportedAudioFormatException("Invalid WAV file: missing WAVE header");
        }
        
        // 查找fmt chunk / Find fmt chunk
        WAVFormatChunk formatChunk = findFormatChunk(buffer);
        
        // 查找data chunk / Find data chunk
        WAVDataChunk dataChunk = findDataChunk(buffer);
        
        // 解析音频数据 / Parse audio data
        IVector<Double> samples = parseAudioSamples(dataChunk.data, formatChunk);
        
        return new AudioData(samples, formatChunk.sampleRate, formatChunk.channels, 
                           formatChunk.bitsPerSample, AudioFormat.WAV);
    }
    
    /**
     * 解析MP3文件 / Parse MP3 file
     *
     * @param fileData MP3文件数据 / MP3 file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果MP3格式不支持 / If MP3 format is not supported
     */
    private static AudioData parseMP3(byte[] fileData) throws UnsupportedAudioFormatException {
        try {
            // Create a ByteArrayInputStream from the file data
            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(fileData);
            PushbackInputStream pushbackInputStream = new PushbackInputStream(byteArrayInputStream, 1024);
            
            // Create a Bitstream to decode the MP3 data
            Bitstream bitstream = new Bitstream(pushbackInputStream);
            
            // Get the first frame to extract header information
            Header header = bitstream.readFrame();
            if (header == null) {
                throw new UnsupportedAudioFormatException("Invalid MP3 file: no frames found");
            }
            
            // Extract audio properties from the header
            int sampleRate = header.frequency();
            int channels = (header.mode() == Header.SINGLE_CHANNEL) ? 1 : 2;
            int bitDepth = 16; // MP3 is typically decoded to 16-bit PCM
            
            // Create a Decoder to decode the MP3 frames
            Decoder decoder = new Decoder();
            
            // Buffer to store all decoded samples
            java.util.List<Double> sampleList = new java.util.ArrayList<>();
            
            // Decode all frames
            Header frameHeader;
            while ((frameHeader = bitstream.readFrame()) != null) {
                // Decode the frame to PCM samples
                SampleBuffer sampleBuffer = (SampleBuffer) decoder.decodeFrame(frameHeader, bitstream);
                
                // Get the decoded PCM samples
                short[] pcmSamples = sampleBuffer.getBuffer();
                int sampleCount = sampleBuffer.getBufferLength();
                
                // Convert short samples to double in the range [-1, 1]
                for (int i = 0; i < sampleCount; i++) {
                    sampleList.add(pcmSamples[i] / 32768.0);
                }
                
                // Close the frame
                bitstream.closeFrame();
            }
            
            // Close the bitstream
            bitstream.close();
            
            // Convert the sample list to an IVector
            int totalSamples = sampleList.size();
            IVector<Double> samples = Linalg.zeros(totalSamples);
            for (int i = 0; i < totalSamples; i++) {
                samples.set(i, sampleList.get(i));
            }
            
            // Return the AudioData object
            return new AudioData(samples, sampleRate, channels, bitDepth, AudioFormat.MP3);
            
        } catch (JavaLayerException e) {
            throw new UnsupportedAudioFormatException("Failed to decode MP3 file: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new UnsupportedAudioFormatException("Unexpected error while parsing MP3 file: " + e.getMessage(), e);
        }
    }
    
    /**
     * 解析RAW音频文件 / Parse RAW audio file
     *
     * @param fileData RAW文件数据 / RAW file data
     * @return 音频数据对象 / Audio data object
     */
    private static AudioData parseRAW(byte[] fileData) {
        // RAW格式假设为16位PCM，单声道，44.1kHz
        // RAW format assumes 16-bit PCM, mono, 44.1kHz
        int samplesCount = fileData.length / 2; // 16位 = 2字节 / 16-bit = 2 bytes
        IVector<Double> samples = Linalg.zeros(samplesCount);
        
        ByteBuffer buffer = ByteBuffer.wrap(fileData);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (int i = 0; i < samplesCount; i++) {
            short sample = buffer.getShort();
            samples.set(i, sample / 32768.0); // 归一化到[-1, 1] / Normalize to [-1, 1]
        }
        
        return new AudioData(samples, DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS, 
                           DEFAULT_BIT_DEPTH, AudioFormat.RAW);
    }
    
    /**
     * 解析FLAC文件 / Parse FLAC file
     *
     * @param fileData FLAC文件数据 / FLAC file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果FLAC格式不支持 / If FLAC format is not supported
     */
    private static AudioData parseFLAC(byte[] fileData) throws UnsupportedAudioFormatException {
        // For now, we'll throw an exception indicating FLAC parsing is not yet implemented
        // A full implementation would require using a library like kopi-java-flac
        throw new UnsupportedAudioFormatException("FLAC parsing not yet implemented. " +
                "To add FLAC support, please implement FLAC decoding using a library like kopi-java-flac.");
    }
    
    /**
     * 解析OGG文件 / Parse OGG file
     *
     * @param fileData OGG文件数据 / OGG file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果OGG格式不支持 / If OGG format is not supported
     */
    private static AudioData parseOGG(byte[] fileData) throws UnsupportedAudioFormatException {
        // For now, we'll throw an exception indicating OGG parsing is not yet implemented
        // A full implementation would require using a library like jorbis
        throw new UnsupportedAudioFormatException("OGG parsing not yet implemented. " +
                "To add OGG support, please implement OGG decoding using a library like jorbis.");
    }
    
    /**
     * 解析AAC文件 / Parse AAC file
     *
     * @param fileData AAC文件数据 / AAC file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果AAC格式不支持 / If AAC format is not supported
     */
    private static AudioData parseAAC(byte[] fileData) throws UnsupportedAudioFormatException {
        // For now, we'll throw an exception indicating AAC parsing is not yet implemented
        // A full implementation would require using a library like JAAD
        throw new UnsupportedAudioFormatException("AAC parsing not yet implemented. " +
                "To add AAC support, please implement AAC decoding using a library like JAAD.");
    }
    
    /**
     * 解析M4A文件 / Parse M4A file
     *
     * @param fileData M4A文件数据 / M4A file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果M4A格式不支持 / If M4A format is not supported
     */
    private static AudioData parseM4A(byte[] fileData) throws UnsupportedAudioFormatException {
        // M4A files typically contain AAC audio, so we can use the same parsing logic
        return parseAAC(fileData);
    }
    
    /**
     * 解析AIFF文件 / Parse AIFF file
     *
     * @param fileData AIFF文件数据 / AIFF file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果AIFF格式不支持 / If AIFF format is not supported
     */
    private static AudioData parseAIFF(byte[] fileData) throws UnsupportedAudioFormatException {
        // For now, we'll throw an exception indicating AIFF parsing is not yet implemented
        // A full implementation would require using Java Sound API or a library like Tritonus
        throw new UnsupportedAudioFormatException("AIFF parsing not yet implemented. " +
                "To add AIFF support, please implement AIFF decoding using Java Sound API or a library like Tritonus.");
    }
    
    /**
     * 解析AU文件 / Parse AU file
     *
     * @param fileData AU文件数据 / AU file data
     * @return 音频数据对象 / Audio data object
     * @throws UnsupportedAudioFormatException 如果AU格式不支持 / If AU format is not supported
     */
    private static AudioData parseAU(byte[] fileData) throws UnsupportedAudioFormatException {
        // For now, we'll throw an exception indicating AU parsing is not yet implemented
        // A full implementation would require using Java Sound API or a library like Tritonus
        throw new UnsupportedAudioFormatException("AU parsing not yet implemented. " +
                "To add AU support, please implement AU decoding using Java Sound API or a library like Tritonus.");
    }
    
    /**
     * 编码为WAV格式 / Encode to WAV format
     *
     * @param audioData 音频数据 / Audio data
     * @return WAV格式字节数据 / WAV format byte data
     */
    private static byte[] encodeWAV(AudioData audioData) {
        IVector<Double> samples = audioData.getSamples();
        int sampleRate = (int) audioData.getSampleRate();
        int channels = audioData.getChannels();
        int bitsPerSample = audioData.getBitDepth();
        
        int bytesPerSample = bitsPerSample / 8;
        int dataSize = samples.length() * bytesPerSample;
        int fileSize = 36 + dataSize;
        
        ByteBuffer buffer = ByteBuffer.allocate(44 + dataSize);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // RIFF头 / RIFF header
        buffer.put("RIFF".getBytes());
        buffer.putInt(fileSize);
        buffer.put("WAVE".getBytes());
        
        // fmt chunk / fmt chunk
        buffer.put("fmt ".getBytes());
        buffer.putInt(16); // fmt chunk size
        buffer.putShort((short) 1); // PCM format
        buffer.putShort((short) channels);
        buffer.putInt(sampleRate);
        buffer.putInt(sampleRate * channels * bytesPerSample); // byte rate
        buffer.putShort((short) (channels * bytesPerSample)); // block align
        buffer.putShort((short) bitsPerSample);
        
        // data chunk / data chunk
        buffer.put("data".getBytes());
        buffer.putInt(dataSize);
        
        // 音频数据 / Audio data
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            short sampleShort = (short) (sample * 32767);
            buffer.putShort(sampleShort);
        }
        
        return buffer.array();
    }
    
    /**
     * 编码为MP3格式 / Encode to MP3 format
     *
     * @param audioData 音频数据 / Audio data
     * @return MP3格式字节数据 / MP3 format byte data
     * @throws UnsupportedAudioFormatException 如果MP3格式不支持 / If MP3 format is not supported
     */
    private static byte[] encodeMP3(AudioData audioData) throws UnsupportedAudioFormatException {
        // MP3 encoding requires a third-party library like LAME
        throw new UnsupportedAudioFormatException("MP3 encoding not yet implemented. " +
                "To add MP3 encoding support, please: \n" +
                "1. Add a dependency on an MP3 encoding library (e.g., LAME)\n" +
                "2. Implement PCM to MP3 compression\n" +
                "Alternatively, export to WAV format instead.");
    }
    
    /**
     * 编码为RAW格式 / Encode to RAW format
     *
     * @param audioData 音频数据 / Audio data
     * @return RAW格式字节数据 / RAW format byte data
     */
    private static byte[] encodeRAW(AudioData audioData) {
        IVector<Double> samples = audioData.getSamples();
        ByteBuffer buffer = ByteBuffer.allocate(samples.length() * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            short sampleShort = (short) (sample * 32767);
            buffer.putShort(sampleShort);
        }
        
        return buffer.array();
    }
    
    /**
     * 编码为FLAC格式 / Encode to FLAC format
     *
     * @param audioData 音频数据 / Audio data
     * @return FLAC格式字节数据 / FLAC format byte data
     * @throws UnsupportedAudioFormatException 如果FLAC格式不支持 / If FLAC format is not supported
     */
    private static byte[] encodeFLAC(AudioData audioData) throws UnsupportedAudioFormatException {
        // FLAC encoding requires a third-party library
        throw new UnsupportedAudioFormatException("FLAC encoding not yet implemented. " +
                "To add FLAC encoding support, please implement FLAC encoding using a library.");
    }
    
    /**
     * 编码为OGG格式 / Encode to OGG format
     *
     * @param audioData 音频数据 / Audio data
     * @return OGG格式字节数据 / OGG format byte data
     * @throws UnsupportedAudioFormatException 如果OGG格式不支持 / If OGG format is not supported
     */
    private static byte[] encodeOGG(AudioData audioData) throws UnsupportedAudioFormatException {
        // OGG encoding requires a third-party library
        throw new UnsupportedAudioFormatException("OGG encoding not yet implemented. " +
                "To add OGG encoding support, please implement OGG encoding using a library.");
    }
    
    /**
     * 编码为AAC格式 / Encode to AAC format
     *
     * @param audioData 音频数据 / Audio data
     * @return AAC格式字节数据 / AAC format byte data
     * @throws UnsupportedAudioFormatException 如果AAC格式不支持 / If AAC format is not supported
     */
    private static byte[] encodeAAC(AudioData audioData) throws UnsupportedAudioFormatException {
        // AAC encoding requires a third-party library
        throw new UnsupportedAudioFormatException("AAC encoding not yet implemented. " +
                "To add AAC encoding support, please implement AAC encoding using a library.");
    }
    
    /**
     * 编码为M4A格式 / Encode to M4A format
     *
     * @param audioData 音频数据 / Audio data
     * @return M4A格式字节数据 / M4A format byte data
     * @throws UnsupportedAudioFormatException 如果M4A格式不支持 / If M4A format is not supported
     */
    private static byte[] encodeM4A(AudioData audioData) throws UnsupportedAudioFormatException {
        // M4A encoding typically uses AAC encoding
        return encodeAAC(audioData);
    }
    
    /**
     * 编码为AIFF格式 / Encode to AIFF format
     *
     * @param audioData 音频数据 / Audio data
     * @return AIFF格式字节数据 / AIFF format byte data
     * @throws UnsupportedAudioFormatException 如果AIFF格式不支持 / If AIFF format is not supported
     */
    private static byte[] encodeAIFF(AudioData audioData) throws UnsupportedAudioFormatException {
        // AIFF encoding requires a specific implementation
        throw new UnsupportedAudioFormatException("AIFF encoding not yet implemented. " +
                "To add AIFF encoding support, please implement AIFF encoding.");
    }
    
    /**
     * 编码为AU格式 / Encode to AU format
     *
     * @param audioData 音频数据 / Audio data
     * @return AU格式字节数据 / AU format byte data
     * @throws UnsupportedAudioFormatException 如果AU格式不支持 / If AU format is not supported
     */
    private static byte[] encodeAU(AudioData audioData) throws UnsupportedAudioFormatException {
        // AU encoding requires a specific implementation
        throw new UnsupportedAudioFormatException("AU encoding not yet implemented. " +
                "To add AU encoding support, please implement AU encoding.");
    }
    
    /**
     * 解析音频样本 / Parse audio samples
     *
     * @param data 音频数据字节 / Audio data bytes
     * @param formatChunk 格式信息 / Format information
     * @return 音频样本向量 / Audio sample vector
     */
    private static IVector<Double> parseAudioSamples(byte[] data, WAVFormatChunk formatChunk) throws UnsupportedAudioFormatException {
        int bytesPerSample = formatChunk.bitsPerSample / 8;
        int samplesCount = data.length / bytesPerSample;
        
        IVector<Double> samples = Linalg.zeros(samplesCount);
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (int i = 0; i < samplesCount; i++) {
            double sample;
            switch (formatChunk.bitsPerSample) {
                case 8:
                    sample = (buffer.get() + 128) / 128.0 - 1.0;
                    break;
                case 16:
                    sample = buffer.getShort() / 32768.0;
                    break;
                case 24:
                    // 24位样本需要特殊处理 / 24-bit samples need special handling
                    byte[] bytes = new byte[3];
                    buffer.get(bytes);
                    int sample24 = ((bytes[2] & 0xFF) << 16) | ((bytes[1] & 0xFF) << 8) | (bytes[0] & 0xFF);
                    if ((sample24 & 0x800000) != 0) {
                        sample24 |= 0xFF000000; // 符号扩展 / Sign extend
                    }
                    sample = sample24 / 8388608.0;
                    break;
                case 32:
                    sample = buffer.getInt() / 2147483648.0;
                    break;
                default:
                    throw new UnsupportedAudioFormatException("Unsupported bit depth: " + formatChunk.bitsPerSample);
            }
            samples.set(i, sample);
        }
        
        return samples;
    }
    
    /**
     * 查找WAV格式块 / Find WAV format chunk
     *
     * @param buffer 字节缓冲区 / Byte buffer
     * @return 格式块信息 / Format chunk information
     * @throws UnsupportedAudioFormatException 如果格式块未找到 / If format chunk not found
     */
    private static WAVFormatChunk findFormatChunk(ByteBuffer buffer) throws UnsupportedAudioFormatException {
        while (buffer.hasRemaining()) {
            byte[] chunkId = new byte[4];
            buffer.get(chunkId);
            String chunkIdStr = new String(chunkId);
            
            if (chunkIdStr.equals("fmt ")) {
                int chunkSize = buffer.getInt();
                byte[] chunkData = new byte[chunkSize];
                buffer.get(chunkData);
                
                ByteBuffer chunkBuffer = ByteBuffer.wrap(chunkData);
                chunkBuffer.order(ByteOrder.LITTLE_ENDIAN);
                
                chunkBuffer.getShort(); // audioFormat
                short channels = chunkBuffer.getShort();
                int sampleRate = chunkBuffer.getInt();
                chunkBuffer.getInt(); // byteRate
                chunkBuffer.getShort(); // blockAlign
                short bitsPerSample = chunkBuffer.getShort();
                
                return new WAVFormatChunk(channels, sampleRate, bitsPerSample);
            } else {
                int chunkSize = buffer.getInt();
                buffer.position(buffer.position() + chunkSize);
            }
        }
        
        throw new UnsupportedAudioFormatException("WAV format chunk not found");
    }
    
    /**
     * 查找WAV数据块 / Find WAV data chunk
     *
     * @param buffer 字节缓冲区 / Byte buffer
     * @return 数据块信息 / Data chunk information
     * @throws UnsupportedAudioFormatException 如果数据块未找到 / If data chunk not found
     */
    private static WAVDataChunk findDataChunk(ByteBuffer buffer) throws UnsupportedAudioFormatException {
        while (buffer.hasRemaining()) {
            byte[] chunkId = new byte[4];
            buffer.get(chunkId);
            String chunkIdStr = new String(chunkId);
            
            if (chunkIdStr.equals("data")) {
                int chunkSize = buffer.getInt();
                byte[] data = new byte[chunkSize];
                buffer.get(data);
                return new WAVDataChunk(data);
            } else {
                int chunkSize = buffer.getInt();
                buffer.position(buffer.position() + chunkSize);
            }
        }
        
        throw new UnsupportedAudioFormatException("WAV data chunk not found");
    }
    
    /**
     * 获取文件扩展名 / Get file extension
     *
     * @param filePath 文件路径 / File path
     * @return 文件扩展名 / File extension
     */
    private static String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        if (lastDot == -1) {
            return "";
        }
        return filePath.substring(lastDot + 1);
    }
    
    /**
     * WAV格式块信息 / WAV format chunk information
     */
    private static class WAVFormatChunk {
        final short channels;
        final int sampleRate;
        final short bitsPerSample;
        
        WAVFormatChunk(short channels, int sampleRate, short bitsPerSample) {
            this.channels = channels;
            this.sampleRate = sampleRate;
            this.bitsPerSample = bitsPerSample;
        }
    }
    
    /**
     * WAV数据块信息 / WAV data chunk information
     */
    private static class WAVDataChunk {
        final byte[] data;
        
        WAVDataChunk(byte[] data) {
            this.data = data;
        }
    }
}