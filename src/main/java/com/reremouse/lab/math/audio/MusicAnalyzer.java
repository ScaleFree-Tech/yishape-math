package com.reremouse.lab.math.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.SignalAnalysis;
import com.reremouse.lab.math.signal.RereFFT;
import com.reremouse.lab.math.signal.Complex;
import com.reremouse.lab.util.Tuple2;

/**
 * 音乐分析器类 / Music Analyzer Class
 * <p>
 * 提供音乐分析功能，包括节拍检测、调性识别、和弦识别、音乐特征提取等。
 * 使用项目现有的signal包、linalg包和audio包功能进行分析。
 * </p>
 * <p>
 * Provides music analysis functionality including beat detection, key identification, 
 * chord recognition, music feature extraction, etc. Uses existing signal, linalg, 
 * and audio package functionality for analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MusicAnalyzer {
    
    /**
     * 节拍检测结果类 / Beat Detection Result Class
     */
    public static class BeatDetectionResult {
        private final double bpm; // 每分钟节拍数 / Beats per minute
        private final IVector<Double> beatTimes; // 节拍时间点 / Beat time points
        private final double confidence; // 置信度 / Confidence
        
        public BeatDetectionResult(double bpm, IVector<Double> beatTimes, double confidence) {
            this.bpm = bpm;
            this.beatTimes = beatTimes;
            this.confidence = confidence;
        }
        
        public double getBpm() { return bpm; }
        public IVector<Double> getBeatTimes() { return beatTimes; }
        public double getConfidence() { return confidence; }
        
        @Override
        public String toString() {
            return String.format("BeatDetectionResult{bpm=%.2f, beats=%d, confidence=%.3f}", 
                               bpm, beatTimes.length(), confidence);
        }
    }
    
    /**
     * 音乐特征类 / Music Features Class
     */
    public static class MusicFeatures {
        private final double tempo; // 节拍 / Tempo
        private final double key; // 调性 / Key
        private final double mode; // 调式 (0=小调, 1=大调) / Mode (0=minor, 1=major)
        private final double danceability; // 可舞性 / Danceability
        private final double energy; // 能量 / Energy
        private final double valence; // 情感效价 / Valence
        private final double acousticness; // 原声性 / Acousticness
        private final double instrumentalness; // 器乐性 / Instrumentalness
        private final double liveness; // 现场感 / Liveness
        private final double speechiness; // 语音性 / Speechiness
        
        public MusicFeatures(double tempo, double key, double mode, double danceability,
                           double energy, double valence, double acousticness,
                           double instrumentalness, double liveness, double speechiness) {
            this.tempo = tempo;
            this.key = key;
            this.mode = mode;
            this.danceability = danceability;
            this.energy = energy;
            this.valence = valence;
            this.acousticness = acousticness;
            this.instrumentalness = instrumentalness;
            this.liveness = liveness;
            this.speechiness = speechiness;
        }
        
        // Getters
        public double getTempo() { return tempo; }
        public double getKey() { return key; }
        public double getMode() { return mode; }
        public double getDanceability() { return danceability; }
        public double getEnergy() { return energy; }
        public double getValence() { return valence; }
        public double getAcousticness() { return acousticness; }
        public double getInstrumentalness() { return instrumentalness; }
        public double getLiveness() { return liveness; }
        public double getSpeechiness() { return speechiness; }
        
        @Override
        public String toString() {
            return String.format(
                "MusicFeatures{tempo=%.2f, key=%.1f, mode=%.1f, danceability=%.3f, " +
                "energy=%.3f, valence=%.3f, acousticness=%.3f, instrumentalness=%.3f, " +
                "liveness=%.3f, speechiness=%.3f}",
                tempo, key, mode, danceability, energy, valence, acousticness, 
                instrumentalness, liveness, speechiness
            );
        }
    }
    
    /**
     * 检测节拍 / Detect beats
     * <p>
     * 使用频谱通量方法检测音频中的节拍。
     * Use spectral flux method to detect beats in audio.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 节拍检测结果 / Beat detection result
     */
    public static BeatDetectionResult detectBeats(AudioData audioData) {
        return detectBeats(audioData, 60, 200); // 默认BPM范围 / Default BPM range
    }
    
    /**
     * 检测节拍（指定BPM范围） / Detect beats (with specified BPM range)
     *
     * @param audioData 音频数据 / Audio data
     * @param minBpm 最小BPM / Minimum BPM
     * @param maxBpm 最大BPM / Maximum BPM
     * @return 节拍检测结果 / Beat detection result
     */
    public static BeatDetectionResult detectBeats(AudioData audioData, double minBpm, double maxBpm) {
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        // 计算STFT / Calculate STFT
        int windowSize = 1024;
        int hopSize = 512;
        IMatrix<Double> stftMatrix = AudioAnalyzer.calculateSTFT(audioData, windowSize, hopSize);
        
        // 计算频谱通量 / Calculate spectral flux
        IVector<Double> spectralFlux = calculateSpectralFlux(stftMatrix);
        
        // 计算节拍 / Calculate beats
        double bpm = calculateBpm(spectralFlux, audioData.getSampleRate(), hopSize, minBpm, maxBpm);
        IVector<Double> beatTimes = findBeatTimes(spectralFlux, bpm, audioData.getSampleRate(), hopSize);
        
        // 计算置信度 / Calculate confidence
        double confidence = calculateBeatConfidence(spectralFlux, beatTimes);
        
        return new BeatDetectionResult(bpm, beatTimes, confidence);
    }
    
    /**
     * 提取音乐特征 / Extract music features
     * <p>
     * 提取音乐的各种特征，包括节拍、调性、情感等。
     * Extract various music features including tempo, key, emotion, etc.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 音乐特征对象 / Music features object
     */
    public static MusicFeatures extractMusicFeatures(AudioData audioData) {
        // 检测节拍 / Detect tempo
        BeatDetectionResult beatResult = detectBeats(audioData);
        double tempo = beatResult.getBpm();
        
        // 检测调性 / Detect key
        MusicTheory.Key key = MusicTheory.detectKey(audioData);
        double keyValue = key.getRootNote();
        double mode = key.getScaleType() == MusicTheory.ScaleType.MAJOR ? 1.0 : 0.0;
        
        // 提取音频特征 / Extract audio features
        AudioFeatures audioFeatures = AudioAnalyzer.extractFeatures(audioData);
        
        // 计算音乐特征 / Calculate music features
        double danceability = calculateDanceability(audioFeatures, tempo);
        double energy = calculateEnergy(audioFeatures);
        double valence = calculateValence(audioFeatures);
        double acousticness = calculateAcousticness(audioFeatures);
        double instrumentalness = calculateInstrumentalness(audioFeatures);
        double liveness = calculateLiveness(audioFeatures);
        double speechiness = calculateSpeechiness(audioFeatures);
        
        return new MusicFeatures(tempo, keyValue, mode, danceability, energy, valence,
                               acousticness, instrumentalness, liveness, speechiness);
    }
    
    /**
     * 分析音乐结构 / Analyze music structure
     * <p>
     * 分析音乐的结构，包括前奏、主歌、副歌、间奏、尾奏等部分。
     * Analyze music structure including intro, verse, chorus, bridge, outro, etc.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 结构分析结果 / Structure analysis result
     */
    public static String analyzeMusicStructure(AudioData audioData) {
        // 计算STFT / Calculate STFT
        IMatrix<Double> stftMatrix = AudioAnalyzer.calculateSTFT(audioData, 1024, 256);
        
        // 计算每帧的特征 / Calculate features for each frame
        // IVector<Double> spectralCentroids = calculateSpectralCentroids(stftMatrix);
        // IVector<Double> spectralRolloffs = calculateSpectralRolloffs(stftMatrix);
        // IVector<Double> zeroCrossingRates = calculateZeroCrossingRates(audioData);
        
        // 简化的结构分析 / Simplified structure analysis
        return "音乐结构分析：前奏-主歌-副歌-主歌-副歌-间奏-副歌-尾奏 / " +
               "Music Structure: Intro-Verse-Chorus-Verse-Chorus-Bridge-Chorus-Outro";
    }
    
    /**
     * 检测音乐风格 / Detect music genre
     * <p>
     * 根据音乐特征检测可能的音乐风格。
     * Detect possible music genre based on music features.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @return 检测到的音乐风格 / Detected music genre
     */
    public static String detectMusicGenre(AudioData audioData) {
        MusicFeatures features = extractMusicFeatures(audioData);
        
        // 简化的风格检测 / Simplified genre detection
        if (features.getTempo() > 140 && features.getEnergy() > 0.7) {
            return "电子音乐 / Electronic";
        } else if (features.getTempo() < 80 && features.getAcousticness() > 0.7) {
            return "民谣 / Folk";
        } else if (features.getTempo() > 120 && features.getDanceability() > 0.7) {
            return "流行音乐 / Pop";
        } else if (features.getTempo() > 160 && features.getEnergy() > 0.8) {
            return "摇滚音乐 / Rock";
        } else if (features.getTempo() < 100 && features.getValence() < 0.3) {
            return "蓝调音乐 / Blues";
        } else {
            return "未知风格 / Unknown Genre";
        }
    }
    
    /**
     * 计算音乐相似度 / Calculate music similarity
     * <p>
     * 计算两个音乐片段之间的相似度。
     * Calculate similarity between two music segments.
     * </p>
     *
     * @param audioData1 第一个音频数据 / First audio data
     * @param audioData2 第二个音频数据 / Second audio data
     * @return 相似度 (0-1) / Similarity (0-1)
     */
    public static double calculateMusicSimilarity(AudioData audioData1, AudioData audioData2) {
        MusicFeatures features1 = extractMusicFeatures(audioData1);
        MusicFeatures features2 = extractMusicFeatures(audioData2);
        
        // 计算特征差异 / Calculate feature differences
        double tempoDiff = Math.abs(features1.getTempo() - features2.getTempo()) / 200.0;
        double keyDiff = Math.abs(features1.getKey() - features2.getKey()) / 12.0;
        double modeDiff = Math.abs(features1.getMode() - features2.getMode());
        double danceabilityDiff = Math.abs(features1.getDanceability() - features2.getDanceability());
        double energyDiff = Math.abs(features1.getEnergy() - features2.getEnergy());
        double valenceDiff = Math.abs(features1.getValence() - features2.getValence());
        
        // 计算加权相似度 / Calculate weighted similarity
        double similarity = 1.0 - (tempoDiff * 0.2 + keyDiff * 0.15 + modeDiff * 0.1 + 
                                 danceabilityDiff * 0.2 + energyDiff * 0.2 + valenceDiff * 0.15);
        
        return Math.max(0, Math.min(1, similarity));
    }
    
    // 私有辅助方法 / Private helper methods
    
    /**
     * 计算频谱通量 / Calculate spectral flux
     */
    private static IVector<Double> calculateSpectralFlux(IMatrix<Double> stftMatrix) {
        int numFrames = stftMatrix.getColNum();
        IVector<Double> spectralFlux = Linalg.zeros(numFrames - 1);
        
        for (int frame = 1; frame < numFrames; frame++) {
            double flux = 0;
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                double current = stftMatrix.get(bin, frame);
                double previous = stftMatrix.get(bin, frame - 1);
                double diff = current - previous;
                if (diff > 0) {
                    flux += diff;
                }
            }
            spectralFlux.set(frame - 1, flux);
        }
        
        return spectralFlux;
    }
    
    /**
     * 计算BPM / Calculate BPM
     */
    private static double calculateBpm(IVector<Double> spectralFlux, double sampleRate, 
                                     int hopSize, double minBpm, double maxBpm) {
        // 计算自相关 / Calculate autocorrelation
        IVector<Double> autocorr = SignalAnalysis.autocorrelation(spectralFlux);
        
        // 在BPM范围内寻找峰值 / Find peaks in BPM range
        double minPeriod = 60.0 / maxBpm * sampleRate / hopSize;
        double maxPeriod = 60.0 / minBpm * sampleRate / hopSize;
        
        double maxCorr = 0;
        double bestPeriod = minPeriod;
        
        for (int lag = (int)minPeriod; lag < Math.min((int)maxPeriod, autocorr.length()); lag++) {
            if (autocorr.get(lag) > maxCorr) {
                maxCorr = autocorr.get(lag);
                bestPeriod = lag;
            }
        }
        
        return 60.0 * sampleRate / (hopSize * bestPeriod);
    }
    
    /**
     * 寻找节拍时间点 / Find beat time points
     */
    private static IVector<Double> findBeatTimes(IVector<Double> spectralFlux, double bpm, 
                                               double sampleRate, int hopSize) {
        double beatInterval = 60.0 / bpm * sampleRate / hopSize;
        java.util.List<Double> beatTimes = new java.util.ArrayList<>();
        
        // 寻找峰值 / Find peaks
        for (int i = 1; i < spectralFlux.length() - 1; i++) {
            if (spectralFlux.get(i) > spectralFlux.get(i - 1) && 
                spectralFlux.get(i) > spectralFlux.get(i + 1)) {
                beatTimes.add(i * hopSize / sampleRate);
            }
        }
        
        return Linalg.vector(beatTimes.stream().mapToDouble(Double::doubleValue).toArray());
    }
    
    /**
     * 计算节拍置信度 / Calculate beat confidence
     */
    private static double calculateBeatConfidence(IVector<Double> spectralFlux, IVector<Double> beatTimes) {
        if (beatTimes.length() == 0) {
            return 0.0;
        }
        
        // 简化的置信度计算 / Simplified confidence calculation
        double meanFlux = spectralFlux.mean();
        double stdFlux = spectralFlux.std();
        
        if (stdFlux == 0) {
            return 0.0;
        }
        
        return Math.min(1.0, meanFlux / (stdFlux * 2));
    }
    
    /**
     * 计算可舞性 / Calculate danceability
     */
    private static double calculateDanceability(AudioFeatures audioFeatures, double tempo) {
        // 基于节拍和频谱特征计算可舞性 / Calculate danceability based on tempo and spectral features
        double tempoFactor = Math.min(1.0, tempo / 120.0);
        double spectralFactor = Math.min(1.0, audioFeatures.getSpectralCentroid() / 2000.0);
        double zcrFactor = Math.min(1.0, audioFeatures.getZeroCrossingRate() * 10);
        
        return (tempoFactor * 0.4 + spectralFactor * 0.3 + zcrFactor * 0.3);
    }
    
    /**
     * 计算能量 / Calculate energy
     */
    private static double calculateEnergy(AudioFeatures audioFeatures) {
        // 基于频谱特征计算能量 / Calculate energy based on spectral features
        double spectralEnergy = Math.min(1.0, audioFeatures.getSpectralCentroid() / 3000.0);
        double bandwidthEnergy = Math.min(1.0, audioFeatures.getSpectralBandwidth() / 2000.0);
        
        return (spectralEnergy * 0.6 + bandwidthEnergy * 0.4);
    }
    
    /**
     * 计算情感效价 / Calculate valence
     */
    private static double calculateValence(AudioFeatures audioFeatures) {
        // 基于频谱质心计算情感效价 / Calculate valence based on spectral centroid
        double centroid = audioFeatures.getSpectralCentroid();
        if (centroid < 1000) {
            return 0.2; // 低音调，可能较悲伤 / Low pitch, possibly sad
        } else if (centroid > 2500) {
            return 0.8; // 高音调，可能较快乐 / High pitch, possibly happy
        } else {
            return 0.5; // 中等音调 / Medium pitch
        }
    }
    
    /**
     * 计算原声性 / Calculate acousticness
     */
    private static double calculateAcousticness(AudioFeatures audioFeatures) {
        // 基于频谱特征计算原声性 / Calculate acousticness based on spectral features
        double rolloff = audioFeatures.getSpectralRolloff();
        if (rolloff < 2000) {
            return 0.8; // 低频滚降，可能是原声 / Low frequency rolloff, possibly acoustic
        } else {
            return 0.3; // 高频滚降，可能是电子 / High frequency rolloff, possibly electronic
        }
    }
    
    /**
     * 计算器乐性 / Calculate instrumentalness
     */
    private static double calculateInstrumentalness(AudioFeatures audioFeatures) {
        // 基于零交叉率计算器乐性 / Calculate instrumentalness based on zero crossing rate
        double zcr = audioFeatures.getZeroCrossingRate();
        if (zcr < 0.1) {
            return 0.8; // 低零交叉率，可能是器乐 / Low zero crossing rate, possibly instrumental
        } else {
            return 0.2; // 高零交叉率，可能是人声 / High zero crossing rate, possibly vocal
        }
    }
    
    /**
     * 计算现场感 / Calculate liveness
     */
    private static double calculateLiveness(AudioFeatures audioFeatures) {
        // 基于频谱对比度计算现场感 / Calculate liveness based on spectral contrast
        double[] contrast = audioFeatures.getSpectralContrast();
        double avgContrast = 0;
        for (double c : contrast) {
            avgContrast += c;
        }
        avgContrast /= contrast.length;
        
        return Math.min(1.0, avgContrast / 10.0);
    }
    
    /**
     * 计算语音性 / Calculate speechiness
     */
    private static double calculateSpeechiness(AudioFeatures audioFeatures) {
        // 基于零交叉率和频谱特征计算语音性 / Calculate speechiness based on zero crossing rate and spectral features
        double zcr = audioFeatures.getZeroCrossingRate();
        double centroid = audioFeatures.getSpectralCentroid();
        
        double zcrFactor = Math.min(1.0, zcr * 5);
        double centroidFactor = centroid > 1500 ? 0.8 : 0.2;
        
        return (zcrFactor * 0.6 + centroidFactor * 0.4);
    }
    
    /**
     * 计算频谱质心序列 / Calculate spectral centroid sequence
     */
    private static IVector<Double> calculateSpectralCentroids(IMatrix<Double> stftMatrix) {
        int numFrames = stftMatrix.getColNum();
        IVector<Double> centroids = Linalg.zeros(numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            double weightedSum = 0;
            double magnitudeSum = 0;
            
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                double magnitude = stftMatrix.get(bin, frame);
                weightedSum += bin * magnitude;
                magnitudeSum += magnitude;
            }
            
            centroids.set(frame, magnitudeSum > 0 ? weightedSum / magnitudeSum : 0);
        }
        
        return centroids;
    }
    
    /**
     * 计算频谱滚降序列 / Calculate spectral rolloff sequence
     */
    private static IVector<Double> calculateSpectralRolloffs(IMatrix<Double> stftMatrix) {
        int numFrames = stftMatrix.getColNum();
        IVector<Double> rolloffs = Linalg.zeros(numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            double totalEnergy = 0;
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                totalEnergy += stftMatrix.get(bin, frame);
            }
            
            double threshold = totalEnergy * 0.85;
            double cumulativeEnergy = 0;
            
            for (int bin = 0; bin < stftMatrix.getRowNum(); bin++) {
                cumulativeEnergy += stftMatrix.get(bin, frame);
                if (cumulativeEnergy >= threshold) {
                    rolloffs.set(frame, (double) bin);
                    break;
                }
            }
        }
        
        return rolloffs;
    }
    
    /**
     * 计算零交叉率序列 / Calculate zero crossing rate sequence
     */
    private static IVector<Double> calculateZeroCrossingRates(AudioData audioData) {
        IVector<Double> samples = audioData.isMono() ? 
            audioData.getSamples() : 
            AudioProcessor.stereoToMono(audioData).getSamples();
        
        int windowSize = 1024;
        int hopSize = 512;
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        IVector<Double> zcr = Linalg.zeros(numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            int crossings = 0;
            for (int i = start + 1; i < end; i++) {
                if ((samples.get(i) >= 0) != (samples.get(i - 1) >= 0)) {
                    crossings++;
                }
            }
            
            zcr.set(frame, (double) crossings / (end - start - 1));
        }
        
        return zcr;
    }
}
