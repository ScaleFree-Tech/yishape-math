package com.yishape.lab.music.processing;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioFormat;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.theory.ChordTheory;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.audio.processing.IAudioProcessor;
import com.yishape.lab.music.theory.ScaleTheory;

import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;

/**
 * Music Transposer
 * <p>
 * Implements music transposition functionality to convert music to different keys.
 * Supports various transposition algorithms including pitch shifting, time stretching,
 * granular synthesis, and phase vocoder methods.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Transposer implements IMusicProcessor {
    
    private static final String NAME = "Music Transposer";
    private static final String VERSION = "1.0.0";
    private static final String DESCRIPTION = "Transposes music to different keys";
    
    // Default parameters
    private static final Map<String, Object> DEFAULT_PARAMETERS = new HashMap<>();
    private static final Set<String> SUPPORTED_PARAMETERS = new HashSet<>();
    
    static {
        DEFAULT_PARAMETERS.put("semitones", 0);           // Semitones to transpose
        DEFAULT_PARAMETERS.put("preserveFormants", true); // Preserve formants
        DEFAULT_PARAMETERS.put("algorithm", "pitchShift"); // Algorithm type
        DEFAULT_PARAMETERS.put("windowSize", 2048);       // Window size
        DEFAULT_PARAMETERS.put("hopSize", 512);           // Hop size
        DEFAULT_PARAMETERS.put("qualityLevel", "high");   // Quality level
        DEFAULT_PARAMETERS.put("fadeInOut", true);        // Apply fade in/out
        DEFAULT_PARAMETERS.put("antiAliasing", true);     // Anti-aliasing filter
        
        SUPPORTED_PARAMETERS.addAll(DEFAULT_PARAMETERS.keySet());
    }
    
    private Map<String, Object> currentParameters;
    private boolean verboseLogging = false;
    private double processingProgress = 0.0;
    private boolean processingCancelled = false;
    private boolean processingPaused = false;
    private Map<String, Object> lastStatistics = new HashMap<>();
    private Map<String, Object> performanceMetrics = new HashMap<>();
    private boolean isReady = true;
    
    /**
     * Constructor
     */
    public Transposer() {
        this.currentParameters = new HashMap<>(DEFAULT_PARAMETERS);
        initializePerformanceMetrics();
    }
    
    // IAudioProcessor methods
    @Override
    public String getName() {
        return NAME;
    }
    
    @Override
    public String getVersion() {
        return VERSION;
    }
    
    @Override
    public void setParameter(String key, Object value) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        currentParameters.put(key, value);
    }
    
    @Override
    public Object getParameter(String key) throws IllegalArgumentException {
        if (key == null) {
            throw new IllegalArgumentException("Parameter key cannot be null");
        }
        return currentParameters.get(key);
    }
    
    @Override
    public void reset() {
        resetParameters();
    }
    
    @Override
    public IAudioProcessor clone() {
        Transposer cloned = new Transposer();
        cloned.currentParameters = new HashMap<>(this.currentParameters);
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
        return DESCRIPTION;
    }
    
    @Override
    public String[] getSupportedParameters() {
        return SUPPORTED_PARAMETERS.toArray(new String[0]);
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        return Collections.unmodifiableMap(DEFAULT_PARAMETERS);
    }
    
    @Override
    public boolean isReady() {
        return isReady;
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) {
            return true; // null parameters means use defaults
        }

        try {
            // Validate semitones
            if (parameters.containsKey("semitones")) {
                Integer semitones = (Integer) parameters.get("semitones");
                if (semitones < -24 || semitones > 24) {
                    return false;
                }
            }
            
            // Validate algorithm
            if (parameters.containsKey("algorithm")) {
                String algorithm = (String) parameters.get("algorithm");
                if (!algorithm.matches("pitchshift|timestretch|granular|phase_vocoder")) {
                    return false;
                }
            }
            
            // Validate window size
            if (parameters.containsKey("windowSize")) {
                Integer windowSize = (Integer) parameters.get("windowSize");
                if (windowSize < 256 || windowSize > 8192) {
                    return false;
                }
            }
            
            // Validate hop size
            if (parameters.containsKey("hopSize")) {
                Integer hopSize = (Integer) parameters.get("hopSize");
                if (hopSize < 64 || hopSize > 4096) {
                    return false;
                }
            }
            
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    @Override
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters != null) {
            if (!validateParameters(parameters)) {
                throw new AudioProcessingException("Invalid parameters provided");
            }
            currentParameters.putAll(parameters);
        }
    }
    
    @Override
    public Map<String, Object> getCurrentParameters() {
        return new HashMap<>(currentParameters);
    }
    
    @Override
    public void resetParameters() {
        currentParameters.clear();
        currentParameters.putAll(DEFAULT_PARAMETERS);
    }
    
    @Override
    public boolean supportsAudioFormat(double sampleRate, int channels, int bitDepth) {
        // Support common audio formats
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
        return process(audioData, currentParameters);
    }
    
    public AudioData process(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }
        
        try {
            processingProgress = 0.0;
            processingCancelled = false;
            
            int semitones = (Integer) parameters.get("semitones");
            boolean preserveFormants = (Boolean) parameters.get("preserveFormants");
            String algorithm = (String) parameters.get("algorithm");
            
            if (verboseLogging) {
                System.out.println("Transposing audio by " + semitones + " semitones using " + algorithm + " algorithm");
            }
            
            AudioData result = performTransposition(audioData, semitones, preserveFormants, algorithm, parameters);
            
            processingProgress = 100.0;
            updateStatistics(audioData, result);
            
            return result;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to transpose audio: " + e.getMessage(), e);
        }
    }
    
    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime) throws AudioProcessingException {
        return processTimeRange(audioData, startTime, endTime, currentParameters);
    }
    
    @Override
    public AudioData processTimeRange(AudioData audioData, double startTime, double endTime, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        if (startTime < 0 || endTime <= startTime || endTime > audioData.getDuration()) {
            throw new AudioProcessingException("Invalid time range");
        }
        
        // Extract specified time segment
        AudioData segment = extractTimeSegment(audioData, startTime, endTime);
        
        // Process audio segment
        AudioData processedSegment = process(segment, parameters);
        
        // Reinsert processed segment into original audio
        return insertProcessedSegment(audioData, processedSegment, startTime, endTime);
    }
    
    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize) throws AudioProcessingException {
        return processStream(audioData, windowSize, hopSize, currentParameters);
    }
    
    @Override
    public AudioData processStream(AudioData audioData, double windowSize, double hopSize, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        if (!validateParameters(parameters)) {
            throw new AudioProcessingException("Invalid parameters provided");
        }
        
        try {
            processingProgress = 0.0;
            processingCancelled = false;
            
            double sampleRate = audioData.getSampleRate();
            int windowSamples = (int) (windowSize * sampleRate);
            int hopSamples = (int) (hopSize * sampleRate);
            
            return performStreamTransposition(audioData, windowSamples, hopSamples, parameters);
            
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to process audio stream: " + e.getMessage(), e);
        }
    }
    
    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray) throws AudioProcessingException {
        return processBatch(audioDataArray, currentParameters);
    }
    
    @Override
    public AudioData[] processBatch(AudioData[] audioDataArray, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioDataArray == null || audioDataArray.length == 0) {
            throw new AudioProcessingException("Audio data array cannot be null or empty");
        }
        
        AudioData[] results = new AudioData[audioDataArray.length];
        
        for (int i = 0; i < audioDataArray.length; i++) {
            if (processingCancelled) {
                break;
            }
            
            results[i] = process(audioDataArray[i], parameters);
            processingProgress = ((double) (i + 1) / audioDataArray.length) * 100.0;
        }
        
        return results;
    }
    
    /**
     * Perform transposition operation
     */
    private AudioData performTransposition(AudioData audioData, int semitones, boolean preserveFormants,
                                         String algorithm, Map<String, Object> parameters) throws AudioProcessingException {
        
        if (semitones == 0) {
            return audioData; // No transposition needed
        }
        
        switch (algorithm.toLowerCase()) {
            case "pitchshift":
                return performPitchShift(audioData, semitones, preserveFormants, parameters);
            case "timestretch":
                return performTimeStretch(audioData, semitones, parameters);
            case "granular":
                return performGranularTransposition(audioData, semitones, parameters);
            case "phase_vocoder":
                return performPhaseVocoderTransposition(audioData, semitones, parameters);
            default:
                throw new AudioProcessingException("Unsupported transposition algorithm: " + algorithm);
        }
    }
    
    /**
     * Pitch shift transposition
     */
    private AudioData performPitchShift(AudioData audioData, int semitones, boolean preserveFormants,
                                       Map<String, Object> parameters) throws AudioProcessingException {
        
        double pitchRatio = Math.pow(2.0, semitones / 12.0);
        int windowSize = (Integer) parameters.get("windowSize");
        int hopSize = (Integer) parameters.get("hopSize");
        
        double[] inputSamples = audioData.getSamples().toDoubleArray();
        double[] outputSamples = new double[inputSamples.length];
        
        // Use PSOLA algorithm for pitch shifting
        performPSOLA(inputSamples, outputSamples, pitchRatio, windowSize, hopSize, preserveFormants);
        
        return new AudioData(Linalg.vector(outputSamples), audioData.getSampleRate(),
                           audioData.getChannels(), outputSamples.length, audioData.getFormat());
    }
    
    /**
     * Time stretch transposition
     */
    private AudioData performTimeStretch(AudioData audioData, int semitones, Map<String, Object> parameters) throws AudioProcessingException {
        
        double timeRatio = Math.pow(2.0, -semitones / 12.0);
        int windowSize = (Integer) parameters.get("windowSize");
        int hopSize = (Integer) parameters.get("hopSize");
        
        double[] inputSamples = audioData.getSamples().toDoubleArray();
        int outputLength = (int) (inputSamples.length * timeRatio);
        double[] outputSamples = new double[outputLength];
        
        // Use phase vocoder for time stretching
        performPhaseVocoder(inputSamples, outputSamples, timeRatio, windowSize, hopSize);
        
        return new AudioData(Linalg.vector(outputSamples), audioData.getSampleRate(), 
                           audioData.getChannels(), outputSamples.length, audioData.getFormat());
    }
    
    /**
     * Granular synthesis transposition
     */
    private AudioData performGranularTransposition(AudioData audioData, int semitones, Map<String, Object> parameters) throws AudioProcessingException {
        
        double pitchRatio = Math.pow(2.0, semitones / 12.0);
        int grainSize = 1024; // Grain size
        int overlap = 512;    // Overlap size
        
        double[] inputSamples = audioData.getSamples().toDoubleArray();
        double[] outputSamples = new double[inputSamples.length];
        
        // Perform granular synthesis transposition
        performGranularSynthesis(inputSamples, outputSamples, pitchRatio, grainSize, overlap);
        
        return new AudioData(Linalg.vector(outputSamples), audioData.getSampleRate(), 
                           audioData.getChannels(), outputSamples.length, audioData.getFormat());
    }
    
    /**
     * Phase vocoder transposition
     */
    private AudioData performPhaseVocoderTransposition(AudioData audioData, int semitones, Map<String, Object> parameters) throws AudioProcessingException {
        
        double pitchRatio = Math.pow(2.0, semitones / 12.0);
        int windowSize = (Integer) parameters.get("windowSize");
        int hopSize = (Integer) parameters.get("hopSize");
        
        double[] inputSamples = audioData.getSamples().toDoubleArray();
        double[] outputSamples = new double[inputSamples.length];
        
        // Use phase vocoder for transposition
        performPhaseVocoderPitchShift(inputSamples, outputSamples, pitchRatio, windowSize, hopSize);
        
        return new AudioData(Linalg.vector(outputSamples), audioData.getSampleRate(), 
                           audioData.getChannels(), outputSamples.length, audioData.getFormat());
    }
    
    /**
     * PSOLA algorithm implementation
     */
    private void performPSOLA(double[] input, double[] output, double pitchRatio, int windowSize, int hopSize, boolean preserveFormants) {
        // Simplified PSOLA implementation
        // Actual implementation requires more complex pitch detection and time-domain processing
        
        for (int i = 0; i < output.length; i++) {
            double sourceIndex = i / pitchRatio;
            
            if (sourceIndex >= 0 && sourceIndex < input.length - 1) {
                // Linear interpolation
                int index = (int) sourceIndex;
                double fraction = sourceIndex - index;
                output[i] = input[index] * (1 - fraction) + input[index + 1] * fraction;
            }
            
            // Update progress
            if (i % 1000 == 0) {
                processingProgress = ((double) i / output.length) * 100.0;
                if (processingCancelled) break;
            }
        }
    }
    
    /**
     * Phase vocoder implementation
     */
    private void performPhaseVocoder(double[] input, double[] output, double timeRatio, int windowSize, int hopSize) {
        // Simplified phase vocoder implementation
        // Actual implementation requires FFT and phase processing
        
        for (int i = 0; i < output.length; i++) {
            double sourceIndex = i / timeRatio;
            
            if (sourceIndex >= 0 && sourceIndex < input.length - 1) {
                int index = (int) sourceIndex;
                double fraction = sourceIndex - index;
                output[i] = input[index] * (1 - fraction) + input[index + 1] * fraction;
            }
            
            if (i % 1000 == 0) {
                processingProgress = ((double) i / output.length) * 100.0;
                if (processingCancelled) break;
            }
        }
    }
    
    /**
     * Granular synthesis implementation
     */
    private void performGranularSynthesis(double[] input, double[] output, double pitchRatio, int grainSize, int overlap) {
        // Simplified granular synthesis implementation
        
        int hopSize = grainSize - overlap;
        
        for (int pos = 0; pos < output.length - grainSize; pos += hopSize) {
            double sourcePos = pos / pitchRatio;
            
            if (sourcePos >= 0 && sourcePos < input.length - grainSize) {
                // Extract and process grain
                for (int i = 0; i < grainSize; i++) {
                    if (pos + i < output.length && sourcePos + i < input.length) {
                        // Apply window function
                        double window = 0.5 * (1 - Math.cos(2 * Math.PI * i / grainSize));
                        output[pos + i] += input[(int)(sourcePos + i)] * window;
                    }
                }
            }
            
            processingProgress = ((double) pos / output.length) * 100.0;
            if (processingCancelled) break;
        }
    }
    
    /**
     * Phase vocoder pitch shift implementation
     */
    private void performPhaseVocoderPitchShift(double[] input, double[] output, double pitchRatio, int windowSize, int hopSize) {
        // Simplified phase vocoder pitch shift implementation
        // Actual implementation requires complex frequency domain processing
        
        for (int i = 0; i < output.length; i++) {
            double sourceIndex = i / pitchRatio;
            
            if (sourceIndex >= 0 && sourceIndex < input.length - 1) {
                int index = (int) sourceIndex;
                double fraction = sourceIndex - index;
                output[i] = input[index] * (1 - fraction) + input[index + 1] * fraction;
            }
            
            if (i % 1000 == 0) {
                processingProgress = ((double) i / output.length) * 100.0;
                if (processingCancelled) break;
            }
        }
    }
    
    /**
     * Stream transposition processing
     */
    private AudioData performStreamTransposition(AudioData audioData, int windowSamples, int hopSamples, Map<String, Object> parameters) throws AudioProcessingException {
        
        double[] inputSamples = audioData.getSamples().toDoubleArray();
        double[] outputSamples = new double[inputSamples.length];
        
        int semitones = (Integer) parameters.get("semitones");
        double pitchRatio = Math.pow(2.0, semitones / 12.0);
        
        // Process in blocks
        for (int pos = 0; pos < inputSamples.length; pos += hopSamples) {
            if (processingCancelled) break;
            
            int blockSize = Math.min(windowSamples, inputSamples.length - pos);
            
            // Process current block
            for (int i = 0; i < blockSize; i++) {
                double sourceIndex = i / pitchRatio;
                if (sourceIndex >= 0 && sourceIndex < blockSize - 1) {
                    int index = (int) sourceIndex;
                    double fraction = sourceIndex - index;
                    outputSamples[pos + i] = inputSamples[pos + index] * (1 - fraction) +
                                           inputSamples[pos + index + 1] * fraction;
                }
            }
            
            processingProgress = ((double) pos / inputSamples.length) * 100.0;
        }
        
        return new AudioData(Linalg.vector(outputSamples), audioData.getSampleRate(), 
                           audioData.getChannels(), outputSamples.length, audioData.getFormat());
    }
    
    /**
     * Extract time segment
     */
    private AudioData extractTimeSegment(AudioData audioData, double startTime, double endTime) {
        double sampleRate = audioData.getSampleRate();
        int startSample = (int) (startTime * sampleRate);
        int endSample = (int) (endTime * sampleRate);
        
        double[] samples = audioData.getSamples().toDoubleArray();
        double[] segmentSamples = new double[endSample - startSample];
        
        System.arraycopy(samples, startSample, segmentSamples, 0, segmentSamples.length);
        
        return new AudioData(Linalg.vector(segmentSamples), sampleRate, 
                           audioData.getChannels(), segmentSamples.length, audioData.getFormat());
    }
    
    /**
     * Insert processed segment
     */
    private AudioData insertProcessedSegment(AudioData originalAudio, AudioData processedSegment, double startTime, double endTime) {
        double sampleRate = originalAudio.getSampleRate();
        int startSample = (int) (startTime * sampleRate);
        int endSample = (int) (endTime * sampleRate);
        
        double[] originalSamples = originalAudio.getSamples().toDoubleArray();
        double[] processedSamples = processedSegment.getSamples().toDoubleArray();
        double[] resultSamples = originalSamples.clone();
        
        // Replace specified segment
        System.arraycopy(processedSamples, 0, resultSamples, startSample,
                        Math.min(processedSamples.length, endSample - startSample));
        
        return new AudioData(Linalg.vector(resultSamples), sampleRate, 
                           originalAudio.getChannels(), resultSamples.length, originalAudio.getFormat());
    }
    
    /**
     * Update statistics
     */
    private void updateStatistics(AudioData input, AudioData output) {
        lastStatistics.clear();
        lastStatistics.put("inputLength", input.getDuration());
        lastStatistics.put("outputLength", output.getDuration());
        lastStatistics.put("inputSamples", input.getSamples().length());
        lastStatistics.put("outputSamples", output.getSamples().length());
        lastStatistics.put("sampleRate", input.getSampleRate());
        lastStatistics.put("channels", input.getChannels());
        lastStatistics.put("semitones", currentParameters.get("semitones"));
        lastStatistics.put("algorithm", currentParameters.get("algorithm"));
    }
    
    /**
     * Initialize performance metrics
     */
    private void initializePerformanceMetrics() {
        performanceMetrics.put("totalProcessingTime", 0L);
        performanceMetrics.put("averageProcessingTime", 0.0);
        performanceMetrics.put("processedSamples", 0L);
        performanceMetrics.put("processedFiles", 0);
    }
    
    // ========== IMusicProcessor 接口实现 / IMusicProcessor Interface Implementation ==========
    
    @Override
    public AudioData transpose(AudioData audioData, int semitones) throws AudioProcessingException {
        Map<String, Object> params = new HashMap<>(currentParameters);
        params.put("semitones", semitones);
        return process(audioData, params);
    }
    
    @Override
    public AudioData harmonize(AudioData audioData, String harmonyType, int voiceCount) throws AudioProcessingException {
        // Transposer doesn't support harmonization
        return audioData;
    }
    
    @Override
    public AudioData quantize(AudioData audioData, double gridSize) throws AudioProcessingException {
        // Transposer doesn't support quantization
        return audioData;
    }
    
    @Override
    public AudioData generateScale(ScaleTheory scale, int rootNote, int octave, double duration) throws AudioProcessingException {
        // Transposer doesn't support scale generation
        double[] samples = new double[(int)(duration * 44100)];
        return new AudioData(Linalg.zeros(samples.length), 44100, 1, samples.length, AudioFormat.WAV);
    }
    
    @Override
    public AudioData generateChord(ChordTheory chord, int rootNote, int octave, double duration) throws AudioProcessingException {
        // Transposer doesn't support chord generation
        double[] samples = new double[(int)(duration * 44100)];
        return new AudioData(Linalg.zeros(samples.length), 44100, 1, samples.length, AudioFormat.WAV);
    }
    
    @Override
    public AudioData applyMusicTransformation(AudioData audioData, String transformation, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }
        
        Map<String, Object> params = new HashMap<>(parameters);
        params.put("operation", transformation);
        return process(audioData, params);
    }
    
    @Override
    public String[] getSupportedMusicTransformations() {
        return new String[]{"transpose"};
    }
    
    @Override
    public String[] getSupportedHarmonyTypes() {
        return new String[]{}; // Transposer doesn't support harmony
    }
    
    @Override
    public String getCurrentQualityLevel() {
        return (String) currentParameters.get("qualityLevel");
    }
    
    @Override
    public double estimateProcessingTime(double audioLength) {
        // Simple estimation: 1.5x the audio length
        return audioLength * 1.5;
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
    public double getMinimumAudioLength() {
        return 0.1; // 100ms minimum
    }
    
    @Override
    public double getMaximumAudioLength() {
        return 3600.0; // 1 hour maximum
    }
    
    @Override
    public double getComplexityEstimate(double audioLength) {
        // Simple complexity estimation
        return audioLength * 0.2;
    }
    
    @Override
    public void warmUp() throws AudioProcessingException {
        // No specific warm-up needed for transposer
    }
    
    @Override
    public void cleanup() {
        // Clean up resources if needed
        currentParameters.clear();
        lastStatistics.clear();
        performanceMetrics.clear();
    }
    
    @Override
    public Map<String, Object> getLastProcessingStatistics() {
        return new HashMap<>(lastStatistics);
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        return new HashMap<>(performanceMetrics);
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
        return new String[]{"transposition"};
    }
    
    @Override
    public boolean supportsProcessingType(String processingType) {
        return "transposition".equals(processingType);
    }
    
    @Override
    public String[] getQualityLevels() {
        return new String[]{"low", "medium", "high"};
    }
    
    @Override
    public void setQualityLevel(String qualityLevel) throws AudioProcessingException {
        if (qualityLevel == null || !("low".equals(qualityLevel) || "medium".equals(qualityLevel) || "high".equals(qualityLevel))) {
            throw new AudioProcessingException("Invalid quality level: " + qualityLevel);
        }
        currentParameters.put("qualityLevel", qualityLevel);
    }
}