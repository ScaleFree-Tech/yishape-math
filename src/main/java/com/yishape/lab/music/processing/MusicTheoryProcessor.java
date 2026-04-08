package com.yishape.lab.music.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.audio.processing.IAudioProcessor;
import com.yishape.lab.music.theory.ChordTheory;
import com.yishape.lab.music.theory.ScaleTheory;

import java.util.*;

/**
 * 音乐理论处理器 / Music Theory Processor
 * <p>
 * 基于音乐理论的音频处理器，提供音阶生成、和弦生成、调性转换等功能。
 * Music theory-based audio processor providing scale generation, chord generation, key transposition, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MusicTheoryProcessor implements IMusicProcessor {
    
    /**
     * 调性枚举 / Key Enumeration
     */
    public enum Key {
        C("C大调", 0), G("G大调", 1), D("D大调", 2), A("A大调", 3), E("E大调", 4), B("B大调", 5),
        F_SHARP("F#大调", 6), C_SHARP("C#大调", 7), F("F大调", -1), B_FLAT("Bb大调", -2),
        E_FLAT("Eb大调", -3), A_FLAT("Ab大调", -4), D_FLAT("Db大调", -5), G_FLAT("Gb大调", -6),
        C_FLAT("Cb大调", -7),
        A_MINOR("a小调", 0), E_MINOR("e小调", 1), B_MINOR("b小调", 2), F_SHARP_MINOR("f#小调", 3),
        C_SHARP_MINOR("c#小调", 4), G_SHARP_MINOR("g#小调", 5), D_SHARP_MINOR("d#小调", 6),
        A_SHARP_MINOR("a#小调", 7), D_MINOR("d小调", -1), G_MINOR("g小调", -2),
        C_MINOR("c小调", -3), F_MINOR("f小调", -4), B_FLAT_MINOR("bb小调", -5), E_FLAT_MINOR("eb小调", -6);
        
        private final String chineseName;
        private final int sharpsFlats;
        
        Key(String chineseName, int sharpsFlats) {
            this.chineseName = chineseName;
            this.sharpsFlats = sharpsFlats;
        }
        
        public String getChineseName() { return chineseName; }
        public int getSharpsFlats() { return sharpsFlats; }
    }
    
    // 默认参数 / Default parameters
    private static final String DEFAULT_OPERATION = "generate_scale";
    private static final int DEFAULT_ROOT_NOTE = 0; // C
    private static final ScaleTheory.ScaleType DEFAULT_SCALE_TYPE = ScaleTheory.ScaleType.MAJOR;
    private static final ChordTheory.ChordType DEFAULT_CHORD_TYPE = ChordTheory.ChordType.MAJOR;
    private static final int DEFAULT_OCTAVE = 4;
    private static final double DEFAULT_DURATION = 1.0;
    private static final double DEFAULT_SAMPLE_RATE = 44100.0;
    
    // 当前参数 / Current parameters
    private final Map<String, Object> parameters = new HashMap<>();
    
    // 处理状态 / Processing state
    private volatile boolean processingCancelled = false;
    private volatile boolean processingPaused = false;
    private volatile double processingProgress = 0.0;
    
    // 日志和统计 / Logging and statistics
    private boolean verboseLogging = false;
    private final Map<String, Object> lastProcessingStatistics = new HashMap<>();
    private long lastProcessingTime = 0;
    
    // 音符频率表 / Note frequency table (A4 = 440Hz)
    private static final double[] NOTE_FREQUENCIES = {
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88
    };
    
    /**
     * 构造函数 / Constructor
     */
    public MusicTheoryProcessor() {
        initializeDefaultParameters();
    }
    
    /**
     * 初始化默认参数 / Initialize default parameters
     */
    private void initializeDefaultParameters() {
        parameters.put("operation", DEFAULT_OPERATION);
        parameters.put("rootNote", DEFAULT_ROOT_NOTE);
        parameters.put("scaleType", DEFAULT_SCALE_TYPE);
        parameters.put("chordType", DEFAULT_CHORD_TYPE);
        parameters.put("octave", DEFAULT_OCTAVE);
        parameters.put("duration", DEFAULT_DURATION);
        parameters.put("sampleRate", DEFAULT_SAMPLE_RATE);
    }
    
    // IAudioProcessor methods
    @Override
    public String getName() {
        return "Music Theory Processor";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        parameters.put(key, value);
    }
    
    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        return parameters.get(key);
    }
    
    @Override
    public void reset() {
        resetParameters();
    }
    
    @Override
    public IAudioProcessor clone() {
        MusicTheoryProcessor cloned = new MusicTheoryProcessor();
        cloned.parameters.putAll(this.parameters);
        return cloned;
    }
    
    @Override
    public boolean supportsFormat(AudioData audioData) {
        if (audioData == null) {
            return false;
        }
        return supportsAudioFormat(audioData.getSampleRate(), audioData.getChannels(), audioData.getBitDepth());
    }
    
    @Override
    public int getLatency() {
        // Return estimated latency in samples
        return 0; // No additional latency in this implementation
    }
    
    // IBaseAudioProcessor methods
    @Override
    public String getDescription() {
        return "基于音乐理论的音频处理器，用于音阶生成、和弦生成和转调 / Music theory-based audio processor for scale generation, chord generation, and transposition";
    }
    
    @Override
    public String[] getSupportedParameters() {
        return new String[]{
            "operation", "rootNote", "scaleType", "chordType", "octave", 
            "duration", "sampleRate", "semitones", "harmonyType", "voiceCount",
            "gridSize", "key", "progression", "chordDuration"
        };
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        return new HashMap<>(parameters);
    }
    
    @Override
    public boolean isReady() {
        return true;
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) return true;
        
        for (String key : parameters.keySet()) {
            if (!Arrays.asList(getSupportedParameters()).contains(key)) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("无效参数 / Invalid parameters");
        }
        this.parameters.putAll(parameters);
    }
    
    @Override
    public Map<String, Object> getCurrentParameters() {
        return new HashMap<>(parameters);
    }
    
    @Override
    public void resetParameters() {
        parameters.clear();
        initializeDefaultParameters();
    }
    
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        return sampleRate > 0 && channels > 0 && bitDepth > 0;
    }
    
    @Override
    public String getStatus() {
        if (processingCancelled) return "cancelled";
        if (processingPaused) return "paused";
        return "ready";
    }
    
    // IMusicProcessor methods
    @Override
    public AudioData process(AudioData audioData) throws AudioProcessingException {
        return process(audioData, parameters);
    }
    
    public AudioData process(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            parameters = this.parameters;
        }
        
        String operation = (String) parameters.getOrDefault("operation", DEFAULT_OPERATION);
        String opLower = operation.toLowerCase(Locale.ROOT);
        boolean needsInputAudio = !"generate_scale".equals(opLower) && !"generate_chord".equals(opLower)
                && !"chord_progression".equals(opLower);
        if (needsInputAudio && audioData == null) {
            throw new AudioProcessingException("音频数据不能为空 / Audio data cannot be null");
        }
        
        try {
            processingProgress = 0.0;
            processingCancelled = false;
            
            long startTime = System.currentTimeMillis();
            AudioData result;
            
            switch (opLower) {
                case "generate_scale":
                    result = generateScaleAudio(parameters);
                    break;
                case "generate_chord":
                    result = generateChordAudio(parameters);
                    break;
                case "transpose":
                    result = transposeAudio(audioData, parameters);
                    break;
                case "harmonize":
                    result = harmonizeAudio(audioData, parameters);
                    break;
                case "quantize":
                    result = quantizeAudio(audioData, parameters);
                    break;
                case "analyze_key":
                    result = analyzeKey(audioData, parameters);
                    break;
                case "chord_progression":
                    result = generateChordProgression(parameters);
                    break;
                default:
                    throw new AudioProcessingException("不支持的操作 / Unsupported operation: " + operation);
            }
            
            long endTime = System.currentTimeMillis();
            lastProcessingTime = endTime - startTime;
            updateProcessingStatistics();
            
            processingProgress = 100.0;
            return result;
            
        } catch (Exception e) {
            throw new AudioProcessingException("音乐理论处理失败 / Music theory processing failed: " + e.getMessage(), e);
        }
    }
    
    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        return processTimeRange(audioData, startTime, endTime, parameters);
    }
    
    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("音频数据不能为空 / Audio data cannot be null");
        }
        if (startTime < 0 || endTime <= startTime || endTime > audioData.getDuration()) {
            throw new AudioProcessingException("无效时间范围 / Invalid time range");
        }
        AudioData segment = extractTimeSegment(audioData, startTime, endTime);
        AudioData processed = process(segment, parameters != null ? parameters : this.parameters);
        return insertProcessedSegment(audioData, processed, startTime, endTime);
    }
    
    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException {
        return processStream(audioData, windowSize, hopSize, parameters);
    }
    
    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("音频数据不能为空 / Audio data cannot be null");
        }
        if (windowSize <= 0 || hopSize <= 0) {
            return process(audioData, parameters);
        }
        double sampleRate = audioData.getSampleRate();
        int channels = audioData.getChannels();
        int windowSamples = (int) Math.round(windowSize * sampleRate);
        int hopSamples = (int) Math.round(hopSize * sampleRate);
        double[] in = audioData.getSamples().toDoubleArray();
        if (windowSamples <= 0 || hopSamples <= 0 || windowSamples > in.length) {
            return process(audioData, parameters);
        }
        Map<String, Object> p = parameters != null ? parameters : this.parameters;
        List<double[]> pieces = new ArrayList<>();
        for (int pos = 0; pos + windowSamples <= in.length; pos += hopSamples) {
            double[] block = new double[windowSamples];
            System.arraycopy(in, pos, block, 0, windowSamples);
            AudioData blockAudio = new AudioData(Linalg.vector(block), sampleRate, channels, windowSamples, audioData.getFormat());
            AudioData proc = process(blockAudio, p);
            pieces.add(proc.getSamples().toDoubleArray());
        }
        if (pieces.isEmpty()) {
            return process(audioData, parameters);
        }
        int totalLen = pieces.stream().mapToInt(a -> a.length).sum();
        double[] merged = new double[totalLen];
        int off = 0;
        for (double[] piece : pieces) {
            System.arraycopy(piece, 0, merged, off, piece.length);
            off += piece.length;
        }
        return new AudioData(Linalg.vector(merged), sampleRate, channels, merged.length, audioData.getFormat());
    }

    private AudioData extractTimeSegment(AudioData audioData, double startTime, double endTime) {
        double sampleRate = audioData.getSampleRate();
        int startSample = (int) (startTime * sampleRate);
        int endSample = (int) (endTime * sampleRate);
        int segmentLength = Math.max(0, endSample - startSample);
        double[] originalSamples = audioData.getSamples().toDoubleArray();
        double[] segmentSamples = new double[segmentLength];
        System.arraycopy(originalSamples, startSample, segmentSamples, 0,
                Math.min(segmentLength, originalSamples.length - startSample));
        return new AudioData(Linalg.vector(segmentSamples), sampleRate, audioData.getChannels(),
                segmentLength, audioData.getFormat());
    }

    private AudioData insertProcessedSegment(AudioData originalAudio, AudioData processedSegment,
            double startTime, double endTime) {
        double sampleRate = originalAudio.getSampleRate();
        int startSample = (int) (startTime * sampleRate);
        double[] originalSamples = originalAudio.getSamples().toDoubleArray();
        double[] processedSamples = processedSegment.getSamples().toDoubleArray();
        double[] outputSamples = new double[originalSamples.length];
        System.arraycopy(originalSamples, 0, outputSamples, 0, originalSamples.length);
        System.arraycopy(processedSamples, 0, outputSamples, startSample,
                Math.min(processedSamples.length, outputSamples.length - startSample));
        return new AudioData(Linalg.vector(outputSamples), sampleRate, originalAudio.getChannels(),
                outputSamples.length, originalAudio.getFormat());
    }
    
    /**
     * 更新处理统计信息 / Update processing statistics
     */
    private void updateProcessingStatistics() {
        lastProcessingStatistics.put("processingTime", lastProcessingTime);
        lastProcessingStatistics.put("processingProgress", processingProgress);
        lastProcessingStatistics.put("cancelled", processingCancelled);
        lastProcessingStatistics.put("paused", processingPaused);
    }
    
    /**
     * 生成音阶音频 / Generate scale audio
     */
    private AudioData generateScaleAudio(Map<String, Object> params) throws AudioProcessingException {
        int rootNote = (Integer) params.getOrDefault("rootNote", DEFAULT_ROOT_NOTE);
        ScaleTheory.ScaleType scaleType = (ScaleTheory.ScaleType) params.getOrDefault("scaleType", DEFAULT_SCALE_TYPE);
        int octave = (Integer) params.getOrDefault("octave", DEFAULT_OCTAVE);
        double duration = (Double) params.getOrDefault("duration", DEFAULT_DURATION);
        double sampleRate = (Double) params.getOrDefault("sampleRate", DEFAULT_SAMPLE_RATE);
        
        try {
            int[] noteNumbers = ScaleTheory.generateScale(rootNote, scaleType);
            double[] frequencies = new double[noteNumbers.length];
            for (int i = 0; i < noteNumbers.length; i++) {
                frequencies[i] = calculateFrequency(noteNumbers[i], octave);
            }
            return generateToneSequence(frequencies, duration, sampleRate);
            
        } catch (Exception e) {
            throw new AudioProcessingException("生成音阶失败 / Failed to generate scale: " + e.getMessage(), e);
        }
    }
    
    /**
     * 生成和弦音频 / Generate chord audio
     */
    private AudioData generateChordAudio(Map<String, Object> params) throws AudioProcessingException {
        int rootNote = (Integer) params.getOrDefault("rootNote", DEFAULT_ROOT_NOTE);
        ChordTheory.ChordType chordType = (ChordTheory.ChordType) params.getOrDefault("chordType", DEFAULT_CHORD_TYPE);
        int octave = (Integer) params.getOrDefault("octave", DEFAULT_OCTAVE);
        double duration = (Double) params.getOrDefault("duration", DEFAULT_DURATION);
        double sampleRate = (Double) params.getOrDefault("sampleRate", DEFAULT_SAMPLE_RATE);
        
        try {
            int[] notes = ChordTheory.generateChord(rootNote, chordType);
            double[] frequencies = new double[notes.length];
            for (int i = 0; i < notes.length; i++) {
                frequencies[i] = calculateFrequency(notes[i], octave);
            }
            return generateChordTones(frequencies, duration, sampleRate);
            
        } catch (Exception e) {
            throw new AudioProcessingException("生成和弦失败 / Failed to generate chord: " + e.getMessage(), e);
        }
    }
    
    /**
     * 转调音频 / Transpose audio
     */
    private AudioData transposeAudio(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        int semitones = (Integer) params.getOrDefault("semitones", 0);
        
        if (semitones == 0) {
            return audioData; // 无需转调
        }
        
        // 简单的音高移位实现
        double pitchRatio = Math.pow(2.0, semitones / 12.0);
        
        // 这里应该实现实际的音高移位算法
        // 为了简化，我们返回原始音频
        return audioData;
    }
    
    /**
     * 和声化音频 / Harmonize audio
     */
    private AudioData harmonizeAudio(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        String harmonyType = (String) params.getOrDefault("harmonyType", "third");
        int voiceCount = (Integer) params.getOrDefault("voiceCount", 3);
        
        // 这里应该实现实际的和声化算法
        // 为了简化，我们返回原始音频
        return audioData;
    }
    
    /**
     * 量化音频 / Quantize audio
     */
    private AudioData quantizeAudio(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        double gridSize = (Double) params.getOrDefault("gridSize", 0.25);
        
        // 简单的量化实现 - 将音频对齐到网格
        if (audioData == null) {
            throw new AudioProcessingException("音频数据不能为空 / Audio data cannot be null");
        }
        
        // 这里应该实现实际的量化逻辑
        return audioData; // 暂时返回原始数据
    }
    
    /**
     * 分析调性 / Analyze key
     */
    private AudioData analyzeKey(AudioData audioData, Map<String, Object> params) throws AudioProcessingException {
        // 这里应该实现调性分析算法
        // 为了简化，我们返回原始音频
        return audioData;
    }
    
    /**
     * 生成和弦进行 / Generate chord progression
     */
    private AudioData generateChordProgression(Map<String, Object> params) throws AudioProcessingException {
        String progression = (String) params.getOrDefault("progression", "I-V-vi-IV");
        Key key = (Key) params.getOrDefault("key", Key.C);
        double chordDuration = (Double) params.getOrDefault("chordDuration", 1.0);
        double sampleRate = (Double) params.getOrDefault("sampleRate", DEFAULT_SAMPLE_RATE);
        
        // 解析和弦进行
        String[] chords = progression.split("-");
        List<AudioData> chordAudios = new ArrayList<>();
        
        for (String chord : chords) {
            // 根据罗马数字和调性生成和弦
            ChordTheory.ChordType chordType = parseRomanNumeral(chord);
            int rootNote = calculateRootNote(chord, key);
            
            Map<String, Object> chordParams = new HashMap<>();
            chordParams.put("rootNote", rootNote);
            chordParams.put("chordType", chordType);
            chordParams.put("duration", chordDuration);
            chordParams.put("sampleRate", sampleRate);
            
            AudioData chordAudio = generateChordAudio(chordParams);
            chordAudios.add(chordAudio);
        }
        
        // 连接所有和弦
        return concatenateAudioData(chordAudios, sampleRate);
    }
    
    /**
     * 解析罗马数字和弦 / Parse Roman numeral chord
     */
    private ChordTheory.ChordType parseRomanNumeral(String romanNumeral) {
        switch (romanNumeral.toUpperCase(Locale.ROOT)) {
            case "I":
            case "IV":
            case "V":
                return ChordTheory.ChordType.MAJOR;
            case "II":
            case "III":
            case "VI":
                return ChordTheory.ChordType.MINOR;
            case "VII":
                return ChordTheory.ChordType.DIMINISHED;
            default:
                return ChordTheory.ChordType.MAJOR;
        }
    }
    
    /**
     * 计算根音 / Calculate root note
     */
    private int calculateRootNote(String romanNumeral, Key key) {
        // 简化实现，基于C大调
        switch (romanNumeral.toUpperCase()) {
            case "I": return 0; // C
            case "II": return 2; // D
            case "III": return 4; // E
            case "IV": return 5; // F
            case "V": return 7; // G
            case "VI": return 9; // A
            case "VII": return 11; // B
            default: return 0;
        }
    }
    
    /**
     * 计算音符频率 / Calculate note frequency
     */
    private double calculateFrequency(int noteNumber, int octave) {
        // A4 = 440Hz, note number 9 in octave 4
        double baseFrequency = NOTE_FREQUENCIES[noteNumber];
        return baseFrequency * Math.pow(2, octave - 4);
    }
    
    /**
     * 生成音调序列 / Generate tone sequence
     */
    private AudioData generateToneSequence(double[] frequencies, double duration, double sampleRate) 
            throws AudioProcessingException {
        double noteDuration = duration / frequencies.length;
        int samplesPerNote = (int) (noteDuration * sampleRate);
        int totalSamples = samplesPerNote * frequencies.length;
        
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        for (int i = 0; i < frequencies.length; i++) {
            int startSample = i * samplesPerNote;
            for (int j = 0; j < samplesPerNote; j++) {
                double t = j / sampleRate;
                double value = 0.5 * Math.sin(2 * Math.PI * frequencies[i] * t);
                samples.set(startSample + j, value);
            }
        }
        
        return new AudioData(samples, sampleRate, 1, totalSamples, AudioFormat.WAV);
    }
    
    /**
     * 生成和弦音调 / Generate chord tones
     */
    private AudioData generateChordTones(double[] frequencies, double duration, double sampleRate) 
            throws AudioProcessingException {
        int totalSamples = (int) (duration * sampleRate);
        IVector<Double> samples = Linalg.zeros(totalSamples);
        
        for (int i = 0; i < totalSamples; i++) {
            double t = i / sampleRate;
            double sample = 0.0;
            
            for (double frequency : frequencies) {
                sample += 0.5 * Math.sin(2 * Math.PI * frequency * t) / frequencies.length;
            }
            
            samples.set(i, sample);
        }
        
        return new AudioData(samples, sampleRate, 1, totalSamples, AudioFormat.WAV);
    }
    
    /**
     * 连接音频数据 / Concatenate audio data
     */
    private AudioData concatenateAudioData(List<AudioData> audioDataList, double sampleRate) 
            throws AudioProcessingException {
        if (audioDataList.isEmpty()) {
            throw new AudioProcessingException("音频数据列表不能为空 / Audio data list cannot be empty");
        }
        
        int totalSamples = audioDataList.stream().mapToInt(audio -> audio.getSamples().size()).sum();
        IVector<Double> concatenatedSamples = Linalg.zeros(totalSamples);
        
        int currentIndex = 0;
        for (AudioData audioData : audioDataList) {
            IVector<Double> samples = audioData.getSamples();
            for (int i = 0; i < samples.size(); i++) {
                concatenatedSamples.set(currentIndex + i, samples.get(i));
            }
            currentIndex += samples.size();
        }
        
        return new AudioData(concatenatedSamples, sampleRate, 1, totalSamples, AudioFormat.WAV);
    }
    
    // ========== IMusicProcessor 接口实现 / IMusicProcessor Interface Implementation ==========
    
    @Override
    public AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>();
        params.put("semitones", semitones);
        params.put("operation", "transpose");
        return process(audioData, params);
    }
    
    @Override
    public AudioData harmonize(AudioData audioData, String harmonyType, int voiceCount) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>();
        params.put("harmonyType", harmonyType);
        params.put("voiceCount", voiceCount);
        params.put("operation", "harmonize");
        return process(audioData, params);
    }
    
    @Override
    public AudioData quantize(AudioData audioData, double gridSize) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>();
        params.put("gridSize", gridSize);
        params.put("operation", "quantize");
        return process(audioData, params);
    }
    
    @Override
    public AudioData generateScale(ScaleTheory.ScaleType scaleType, int rootNote, int octave, double duration) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>();
        params.put("scaleType", scaleType);
        params.put("rootNote", rootNote);
        params.put("octave", octave);
        params.put("duration", duration);
        params.put("operation", "generate_scale");
        return process(null, params);
    }
    
    @Override
    public AudioData generateChord(ChordTheory.ChordType chordType, int rootNote, int octave, double duration) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>();
        params.put("chordType", chordType);
        params.put("rootNote", rootNote);
        params.put("octave", octave);
        params.put("duration", duration);
        params.put("operation", "generate_chord");
        return process(null, params);
    }
    
    @Override
    public AudioData applyMusicTransformation(AudioData audioData, String transformation, Map<String, Object> parameters) 
            throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("音频数据不能为空 / Audio data cannot be null");
        }
        
        Map<String, Object> params = new HashMap<>(parameters);
        params.put("operation", transformation);
        return process(audioData, params);
    }
    
    @Override
    public String[] getSupportedMusicTransformations() {
        return new String[]{
            "transpose", "harmonize", "quantize", "generate_scale", 
            "generate_chord", "analyze_key", "chord_progression"
        };
    }
    
    @Override
    public String[] getSupportedHarmonyTypes() {
        return new String[]{"third", "fourth", "fifth", "sixth"};
    }
    
    @Override
    public String getCurrentQualityLevel() {
        return (String) parameters.getOrDefault("qualityLevel", "medium");
    }
    
    @Override
    public double estimateProcessingTime(double audioLength) {
        return audioLength * 0.05; // Estimate 5% of audio length
    }
    
    @Override
    public double getProcessingProgress() {
        return processingProgress;
    }
    
    @Override
    public boolean cancelProcessing() {
        processingCancelled = true;
        return true;
    }
    
    @Override
    public boolean isProcessingCancelled() {
        return processingCancelled;
    }
    
    @Override
    public boolean pauseProcessing() {
        processingPaused = true;
        return true;
    }
    
    @Override
    public boolean resumeProcessing() {
        processingPaused = false;
        return true;
    }
    
    @Override
    public boolean isProcessingPaused() {
        return processingPaused;
    }
    
    // ========== IAdvancedAudioProcessor 接口实现 / IAdvancedAudioProcessor Interface Implementation ==========
    
    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray) throws AudioProcessingException {
        AudioData[] results = new AudioData[audioDataArray.length];
        for (int i = 0; i < audioDataArray.length; i++) {
            results[i] = process(audioDataArray[i]);
        }
        return results;
    }
    
    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray, Map<String, Object> parameters) throws AudioProcessingException {
        AudioData[] results = new AudioData[audioDataArray.length];
        for (int i = 0; i < audioDataArray.length; i++) {
            results[i] = process(audioDataArray[i], parameters);
        }
        return results;
    }
    
    @Override
    public double getMinimumAudioLength() {
        return 0.1; // 0.1 seconds
    }
    
    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour
    }
    
    @Override
    public double getComplexityEstimate(double audioLength) {
        return audioLength * 0.1; // Linear complexity
    }
    
    @Override
    public void warmUp() throws AudioProcessingException {
        // No specific warm-up needed
    }
    
    @Override
    public void cleanup() {
        // Clean up resources if needed
        parameters.clear();
        lastProcessingStatistics.clear();
    }
    
    @Override
    public Map<String, Object> getLastProcessingStatistics() {
        return new HashMap<>(lastProcessingStatistics);
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("processingTime", lastProcessingTime);
        metrics.put("memoryUsage", 0.0);
        metrics.put("cpuUsage", 0.0);
        return metrics;
    }
    
    @Override
    public void setVerboseLogging(boolean enabled) {
        verboseLogging = enabled;
    }
    
    @Override
    public boolean isVerboseLoggingEnabled() {
        return verboseLogging;
    }
    
    @Override
    public String[] getSupportedProcessingTypes() {
        return new String[]{"scale_generation", "chord_generation", "transposition", "harmonization"};
    }
    
    @Override
    public boolean supportsProcessingType(String processingType) {
        return Arrays.asList(getSupportedProcessingTypes()).contains(processingType);
    }
    
    @Override
    public String[] getQualityLevels() {
        return new String[]{"low", "medium", "high"};
    }
    
    @Override
    public void setQualityLevel(String qualityLevel) throws AudioProcessingException {
        parameters.put("qualityLevel", qualityLevel);
    }
    
    // ========== 工具方法 / Utility Methods ==========
    
    /**
     * 获取支持的音阶类型 / Get supported scale types
     */
    public static ScaleTheory.ScaleType[] getSupportedScaleTypes() {
        return ScaleTheory.ScaleType.values();
    }
    
    /**
     * 获取支持的和弦类型 / Get supported chord types
     */
    public static ChordTheory.ChordType[] getSupportedChordTypes() {
        return ChordTheory.ChordType.values();
    }
    
    /**
     * 获取支持的调性 / Get supported keys
     */
    public static Key[] getSupportedKeys() {
        return Key.values();
    }
    
    /**
     * 根据名称获取音阶类型 / Get scale type by name
     */
    public static ScaleTheory.ScaleType getScaleTypeByName(String name) {
        for (ScaleTheory.ScaleType type : ScaleTheory.ScaleType.values()) {
            if (type.name().equalsIgnoreCase(name) || type.getChineseName().equals(name)
                    || type.getEnglishName().equalsIgnoreCase(name)) {
                return type;
            }
        }
        return null;
    }
    
    /**
     * 根据名称获取和弦类型 / Get chord type by name
     */
    public static ChordTheory.ChordType getChordTypeByName(String name) {
        for (ChordTheory.ChordType type : ChordTheory.ChordType.values()) {
            if (type.name().equalsIgnoreCase(name) || type.getChineseName().equals(name)
                    || type.getEnglishName().equalsIgnoreCase(name)) {
                return type;
            }
        }
        return null;
    }
    
    /**
     * 根据名称获取调性 / Get key by name
     */
    public static Key getKeyByName(String name) {
        for (Key key : Key.values()) {
            if (key.name().equalsIgnoreCase(name) || key.getChineseName().equals(name)) {
                return key;
            }
        }
        return null;
    }
}