package com.reremouse.lab.audio.music;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.MusicAnalyzer;
import com.reremouse.lab.audio.MusicTheory;
import com.reremouse.lab.audio.core.AbstractAudioAnalyzer;
import com.reremouse.lab.audio.core.IMusicAnalyzer;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

/**
 * 集成音乐分析器 / Integrated Music Analyzer
 * <p>
 * 整合现有MusicAnalyzer功能到接口架构中，实现IMusicAnalyzer接口。
 * 使用Strategy模式提供可插拔的音乐分析算法实现。
 * </p>
 * <p>
 * Integrates existing MusicAnalyzer functionality into interface architecture,
 * implementing IMusicAnalyzer interface. Uses Strategy pattern to provide
 * pluggable music analysis algorithm implementations.
 * </p>
 *
 * @author Qoder AI
 * @version 1.0
 * @since 1.0
 */
public class IntegratedMusicAnalyzer extends AbstractAudioAnalyzer implements IMusicAnalyzer {
    
    /** 音乐分析参数 / Music analysis parameters */
    private static final String[] MUSIC_FEATURE_TYPES = {
        "tempo", "key", "mode", "danceability", "energy", "valence",
        "acousticness", "instrumentalness", "liveness", "speechiness",
        "beats", "chords", "music_structure", "genre", "emotion", "complexity"
    };
    
    /** 默认音乐分析参数 / Default music analysis parameters */
    private static final String PARAM_MIN_BPM = "min_bpm";
    private static final String PARAM_MAX_BPM = "max_bpm";
    private static final String PARAM_BEAT_SENSITIVITY = "beat_sensitivity";
    private static final String PARAM_CHORD_THRESHOLD = "chord_threshold";
    private static final String PARAM_KEY_CONFIDENCE = "key_confidence";
    
    /**
     * 构造函数 / Constructor
     */
    public IntegratedMusicAnalyzer() {
        super("IntegratedMusicAnalyzer", MUSIC_FEATURE_TYPES);
    }
    
    @Override
    protected void initializeDefaultParameters() {
        super.initializeDefaultParameters();
        // 音乐分析特定参数 / Music analysis specific parameters
        parameters.put(PARAM_MIN_BPM, 60.0);
        parameters.put(PARAM_MAX_BPM, 200.0);
        parameters.put(PARAM_BEAT_SENSITIVITY, 0.5);
        parameters.put(PARAM_CHORD_THRESHOLD, 0.3);
        parameters.put(PARAM_KEY_CONFIDENCE, 0.7);
    }
    
    @Override
    protected IVector<Double> doExtractFeatures(AudioData audioData) throws AudioProcessingException {
        try {
            // 提取音乐特征 / Extract music features
            MusicAnalyzer.MusicFeatures musicFeatures = MusicAnalyzer.extractMusicFeatures(audioData);
            
            // 转换为向量格式 / Convert to vector format
            double[] featureArray = {
                musicFeatures.getTempo(),
                musicFeatures.getKey(),
                musicFeatures.getMode(),
                musicFeatures.getDanceability(),
                musicFeatures.getEnergy(),
                musicFeatures.getValence(),
                musicFeatures.getAcousticness(),
                musicFeatures.getInstrumentalness(),
                musicFeatures.getLiveness(),
                musicFeatures.getSpeechiness()
            };
            
            return com.reremouse.lab.math.linalg.Linalg.vector(featureArray);
            
        } catch (Exception e) {
            throw new AudioProcessingException("Music feature extraction failed", e);
        }
    }
    
    @Override
    protected Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData audioData) throws AudioProcessingException {
        try {
            // 使用现有的音频分析器计算频谱 / Use existing audio analyzer to calculate spectrum
            return com.reremouse.lab.audio.AudioAnalyzer.calculateSpectrum(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Spectrum calculation failed", e);
        }
    }
    
    @Override
    public MusicAnalyzer.BeatDetectionResult detectBeats(AudioData audioData) throws AudioProcessingException {
        try {
            return MusicAnalyzer.detectBeats(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Beat detection failed", e);
        }
    }
    
    @Override
    public MusicAnalyzer.BeatDetectionResult detectBeats(AudioData audioData, double minBpm, double maxBpm) throws AudioProcessingException {
        try {
            return MusicAnalyzer.detectBeats(audioData, minBpm, maxBpm);
        } catch (Exception e) {
            throw new AudioProcessingException("Beat detection with BPM range failed", e);
        }
    }
    
    @Override
    public MusicTheory.Key detectKey(AudioData audioData) throws AudioProcessingException {
        try {
            return MusicTheory.detectKey(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Key detection failed", e);
        }
    }
    
    @Override
    public MusicTheory.Chord detectChord(AudioData audioData) throws AudioProcessingException {
        try {
            return MusicTheory.detectChord(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Chord detection failed", e);
        }
    }
    
    @Override
    public MusicAnalyzer.MusicFeatures extractMusicFeatures(AudioData audioData) throws AudioProcessingException {
        try {
            return MusicAnalyzer.extractMusicFeatures(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Music features extraction failed", e);
        }
    }
    
    @Override
    public String detectMusicGenre(AudioData audioData) throws AudioProcessingException {
        try {
            return MusicAnalyzer.detectMusicGenre(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Music genre detection failed", e);
        }
    }
    
    @Override
    public String analyzeMusicStructure(AudioData audioData) throws AudioProcessingException {
        try {
            return MusicAnalyzer.analyzeMusicStructure(audioData);
        } catch (Exception e) {
            throw new AudioProcessingException("Music structure analysis failed", e);
        }
    }
    
    @Override
    public double calculateMusicSimilarity(AudioData audioData1, AudioData audioData2) throws AudioProcessingException {
        try {
            return MusicAnalyzer.calculateMusicSimilarity(audioData1, audioData2);
        } catch (Exception e) {
            throw new AudioProcessingException("Music similarity calculation failed", e);
        }
    }
    
    @Override
    public String detectMusicEmotion(AudioData audioData) throws AudioProcessingException {
        try {
            // 基于音乐特征检测情感 / Detect emotion based on music features
            MusicAnalyzer.MusicFeatures features = extractMusicFeatures(audioData);
            
            double valence = features.getValence();
            double energy = features.getEnergy();
            
            if (valence > 0.6 && energy > 0.6) {
                return "快乐/兴奋 / Happy/Excited";
            } else if (valence > 0.6 && energy < 0.4) {
                return "平静/放松 / Calm/Relaxed";
            } else if (valence < 0.4 && energy > 0.6) {
                return "愤怒/激动 / Angry/Aggressive";
            } else if (valence < 0.4 && energy < 0.4) {
                return "悲伤/忧郁 / Sad/Melancholic";
            } else {
                return "中性 / Neutral";
            }
        } catch (Exception e) {
            throw new AudioProcessingException("Music emotion detection failed", e);
        }
    }
    
    @Override
    public double analyzeMusicComplexity(AudioData audioData) throws AudioProcessingException {
        try {
            // 计算音乐复杂度 / Calculate music complexity
            MusicAnalyzer.MusicFeatures features = extractMusicFeatures(audioData);
            
            // 基于多个特征计算复杂度 / Calculate complexity based on multiple features
            double tempoComplexity = Math.min(1.0, features.getTempo() / 180.0);
            double harmonicComplexity = 1.0 - features.getAcousticness(); // 电子音乐通常更复杂
            double rhythmicComplexity = features.getDanceability();
            
            return (tempoComplexity * 0.3 + harmonicComplexity * 0.4 + rhythmicComplexity * 0.3);
        } catch (Exception e) {
            throw new AudioProcessingException("Music complexity analysis failed", e);
        }
    }
    
    @Override
    public String[] detectInstruments(AudioData audioData) throws AudioProcessingException {
        try {
            // 简化的乐器检测 / Simplified instrument detection
            MusicAnalyzer.MusicFeatures features = extractMusicFeatures(audioData);
            
            java.util.List<String> instruments = new java.util.ArrayList<>();
            
            if (features.getAcousticness() > 0.7) {
                instruments.add("吉他 / Guitar");
                instruments.add("钢琴 / Piano");
            }
            
            if (features.getEnergy() > 0.8) {
                instruments.add("鼓 / Drums");
            }
            
            if (features.getInstrumentalness() < 0.3) {
                instruments.add("人声 / Vocals");
            }
            
            if (instruments.isEmpty()) {
                instruments.add("未知乐器 / Unknown Instruments");
            }
            
            return instruments.toArray(new String[0]);
        } catch (Exception e) {
            throw new AudioProcessingException("Instrument detection failed", e);
        }
    }
    
    @Override
    public IVector<Double> analyzeMusicDynamics(AudioData audioData) throws AudioProcessingException {
        try {
            // 分析音乐动态变化 / Analyze music dynamics
            int windowSize = getWindowSize();
            int hopSize = getHopSize();
            IVector<Double> samples = audioData.getSamples();
            
            int numFrames = (samples.length() - windowSize) / hopSize + 1;
            double[] dynamics = new double[numFrames];
            
            for (int i = 0; i < numFrames; i++) {
                int start = i * hopSize;
                int end = Math.min(start + windowSize, samples.length());
                
                // 计算RMS能量 / Calculate RMS energy
                double energy = 0.0;
                for (int j = start; j < end; j++) {
                    double sample = samples.get(j);
                    energy += sample * sample;
                }
                energy = Math.sqrt(energy / (end - start));
                dynamics[i] = energy;
            }
            
            return com.reremouse.lab.math.linalg.Linalg.vector(dynamics);
        } catch (Exception e) {
            throw new AudioProcessingException("Music dynamics analysis failed", e);
        }
    }
    
    @Override
    public void setMusicParameter(String parameterName, Object value) throws IllegalArgumentException {
        setParameter(parameterName, value);
    }
    
    @Override
    public Object getMusicParameter(String parameterName) throws IllegalArgumentException {
        return getParameter(parameterName);
    }
    
    @Override
    public String[] getSupportedMusicFeatureTypes() {
        return MUSIC_FEATURE_TYPES.clone();
    }
    
    @Override
    public boolean supportsMusicFeatureType(String featureType) {
        if (featureType == null) return false;
        
        for (String supportedType : MUSIC_FEATURE_TYPES) {
            if (supportedType.equalsIgnoreCase(featureType)) {
                return true;
            }
        }
        return false;
    }
    
    @Override
    protected int getDefaultFeatureDimension(String featureType) {
        // 为音乐特征提供专门的维度信息 / Provide specialized dimension info for music features
        switch (featureType.toLowerCase()) {
            case "tempo":
            case "key":
            case "mode":
            case "danceability":
            case "energy":
            case "valence":
            case "acousticness":
            case "instrumentalness":
            case "liveness":
            case "speechiness":
            case "complexity":
                return 1;
            case "beats":
                return -1; // 可变长度 / Variable length
            case "chords":
                return 4; // 根音 + 和弦类型编码 / Root note + chord type encoding
            case "music_structure":
                return -1; // 可变长度 / Variable length
            case "genre":
            case "emotion":
                return 1; // 分类标签 / Category label
            default:
                return super.getDefaultFeatureDimension(featureType);
        }
    }
    
    @Override
    protected void validateParameter(String key, Object value) throws IllegalArgumentException {
        switch (key) {
            case PARAM_MIN_BPM:
            case PARAM_MAX_BPM:
                if (!(value instanceof Number) || ((Number) value).doubleValue() <= 0) {
                    throw new IllegalArgumentException(key + " must be a positive number");
                }
                break;
            case PARAM_BEAT_SENSITIVITY:
            case PARAM_CHORD_THRESHOLD:
            case PARAM_KEY_CONFIDENCE:
                if (!(value instanceof Number)) {
                    throw new IllegalArgumentException(key + " must be a number");
                }
                double val = ((Number) value).doubleValue();
                if (val < 0.0 || val > 1.0) {
                    throw new IllegalArgumentException(key + " must be between 0.0 and 1.0");
                }
                break;
            default:
                super.validateParameter(key, value);
                break;
        }
    }
}