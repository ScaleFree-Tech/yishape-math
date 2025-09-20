package com.reremouse.lab.audio.features;

import com.reremouse.lab.audio.AudioData;
import com.reremouse.lab.audio.AudioProcessor;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.RereFFT;
import com.reremouse.lab.math.signal.Complex;
import com.reremouse.lab.util.Tuple2;

/**
 * 高级音频特征提取类 / Advanced Audio Features Extraction Class
 * <p>
 * 提供高级音频特征提取功能，包括色度、音调高度、频谱对比度、谐波感知音调阶级描述等。
 * 这些特征在音乐信息检索、音频分类、音乐推荐等应用中非常重要。
 * </p>
 * <p>
 * Provides advanced audio feature extraction including chroma, tonnetz, spectral contrast, 
 * harmonic pitch class profiles, etc. These features are crucial for music information retrieval, 
 * audio classification, music recommendation applications.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AdvancedAudioFeatures {
    
    /** 标准A4音高频率 / Standard A4 pitch frequency */
    private static final double A4_FREQUENCY = 440.0;
    
    /** 十二平均律中的半音比例 / Semitone ratio in equal temperament */
    private static final double SEMITONE_RATIO = Math.pow(2.0, 1.0/12.0);
    
    /** 色度音名 / Chroma note names */
    private static final String[] CHROMA_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    /** Tonnetz坐标系的基础频率比例 / Base frequency ratios for Tonnetz coordinate system */
    private static final double[][] TONNETZ_VECTORS = {
        {0.0, 1.0},           // 纯五度 / Perfect fifth
        {Math.sqrt(3)/2, 0.5}, // 大三度 / Major third
        {-Math.sqrt(3)/2, 0.5} // 小三度 / Minor third
    };
    
    /**
     * 色度特征提取 / Chroma Feature Extraction
     * <p>
     * 提取音频的12维色度特征，表示12个半音的能量分布。
     * 色度特征对音调变换具有不变性，广泛用于和弦识别和调性分析。
     * </p>
     * <p>
     * Extract 12-dimensional chroma features representing energy distribution of 12 semitones.
     * Chroma features are invariant to pitch transposition, widely used for chord recognition and key analysis.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return 色度特征矩阵 (12 x 帧数) / Chroma feature matrix (12 x frames)
     * @throws AudioProcessingException 处理过程中发生错误 / Error during processing
     */
    public static IMatrix<Double> extractChromaFeatures(AudioData audioData, int windowSize, int hopSize) 
            throws AudioProcessingException {
        
        // 转换为单声道 / Convert to mono
        AudioData monoAudio = audioData.isMono() ? audioData : AudioProcessor.stereoToMono(audioData);
        IVector<Double> samples = monoAudio.getSamples();
        
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        IMatrix<Double> chromaMatrix = Linalg.zeros(12, numFrames);
        
        // 创建汉宁窗 / Create Hanning window
        IVector<Double> window = createHanningWindow(windowSize);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            // 提取窗口信号 / Extract windowed signal
            IVector<Double> windowedSignal = extractWindow(samples, start, end, window);
            
            // 计算FFT / Calculate FFT
            Complex[] fftResult = RereFFT.fft(convertToComplex(windowedSignal));
            
            // 计算色度特征 / Calculate chroma features
            IVector<Double> chromaVector = calculateChromaFromFFT(fftResult, audioData.getSampleRate());
            
            // 存储到矩阵中 / Store in matrix
            for (int chroma = 0; chroma < 12; chroma++) {
                chromaMatrix.set(chroma, frame, chromaVector.get(chroma));
            }
        }
        
        return chromaMatrix;
    }
    
    /**
     * Tonnetz特征提取 / Tonnetz Feature Extraction
     * <p>
     * 基于音调网络(Tonnetz)理论提取音高特征，将色度特征映射到二维调性空间。
     * Tonnetz特征能够很好地表示音乐的调性结构和和声关系。
     * </p>
     * <p>
     * Extract pitch features based on Tonnetz (tonal network) theory, mapping chroma features to 2D tonal space.
     * Tonnetz features can well represent tonal structure and harmonic relationships in music.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return Tonnetz特征矩阵 (6 x 帧数) / Tonnetz feature matrix (6 x frames)
     * @throws AudioProcessingException 处理过程中发生错误 / Error during processing
     */
    public static IMatrix<Double> extractTonnetzFeatures(AudioData audioData, int windowSize, int hopSize) 
            throws AudioProcessingException {
        
        // 首先提取色度特征 / First extract chroma features
        IMatrix<Double> chromaMatrix = extractChromaFeatures(audioData, windowSize, hopSize);
        
        int numFrames = chromaMatrix.getColNum();
        IMatrix<Double> tonnetzMatrix = Linalg.zeros(6, numFrames);
        
        for (int frame = 0; frame < numFrames; frame++) {
            // 获取当前帧的色度向量 / Get chroma vector for current frame
            IVector<Double> chromaVector = Linalg.zeros(12);
            for (int chroma = 0; chroma < 12; chroma++) {
                chromaVector.set(chroma, chromaMatrix.get(chroma, frame));
            }
            
            // 计算Tonnetz坐标 / Calculate Tonnetz coordinates
            IVector<Double> tonnetzVector = calculateTonnetzFromChroma(chromaVector);
            
            // 存储到矩阵中 / Store in matrix
            for (int dim = 0; dim < 6; dim++) {
                tonnetzMatrix.set(dim, frame, tonnetzVector.get(dim));
            }
        }
        
        return tonnetzMatrix;
    }
    
    /**
     * 增强的频谱对比度特征 / Enhanced Spectral Contrast Features
     * <p>
     * 计算多个频段的频谱对比度，捕获音频的谐波和非谐波成分的相对强度。
     * 频谱对比度特征对音乐的音色和质感有很好的描述能力。
     * </p>
     * <p>
     * Calculate spectral contrast in multiple frequency bands, capturing relative intensity 
     * of harmonic and non-harmonic components. Spectral contrast features have good 
     * descriptive ability for timbre and texture of music.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @param numBands 频段数量 / Number of bands
     * @return 频谱对比度矩阵 (频段数 x 帧数) / Spectral contrast matrix (bands x frames)
     * @throws AudioProcessingException 处理过程中发生错误 / Error during processing
     */
    public static IMatrix<Double> extractSpectralContrastFeatures(AudioData audioData, int windowSize, 
            int hopSize, int numBands) throws AudioProcessingException {
        
        // 转换为单声道 / Convert to mono
        AudioData monoAudio = audioData.isMono() ? audioData : AudioProcessor.stereoToMono(audioData);
        IVector<Double> samples = monoAudio.getSamples();
        
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        IMatrix<Double> contrastMatrix = Linalg.zeros(numBands, numFrames);
        
        // 创建汉宁窗 / Create Hanning window
        IVector<Double> window = createHanningWindow(windowSize);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            // 提取窗口信号 / Extract windowed signal
            IVector<Double> windowedSignal = extractWindow(samples, start, end, window);
            
            // 计算FFT / Calculate FFT
            Complex[] fftResult = RereFFT.fft(convertToComplex(windowedSignal));
            
            // 计算幅度谱 / Calculate magnitude spectrum
            IVector<Double> magnitudeSpectrum = calculateMagnitudeSpectrum(fftResult);
            
            // 计算频谱对比度 / Calculate spectral contrast
            IVector<Double> contrastVector = calculateSpectralContrast(magnitudeSpectrum, numBands);
            
            // 存储到矩阵中 / Store in matrix
            for (int band = 0; band < numBands; band++) {
                contrastMatrix.set(band, frame, contrastVector.get(band));
            }
        }
        
        return contrastMatrix;
    }
    
    /**
     * 谐波感知音调阶级描述 (HPCP) / Harmonic Pitch Class Profiles (HPCP)
     * <p>
     * 提取谐波感知的音调阶级描述，这是色度特征的增强版本，
     * 考虑了谐波结构，对音调识别和音乐分析更加准确。
     * </p>
     * <p>
     * Extract Harmonic Pitch Class Profiles, which is an enhanced version of chroma features
     * considering harmonic structure, more accurate for pitch recognition and music analysis.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @param numBins 音调分类数量 / Number of pitch class bins
     * @return HPCP特征矩阵 / HPCP feature matrix
     * @throws AudioProcessingException 处理过程中发生错误 / Error during processing
     */
    public static IMatrix<Double> extractHPCPFeatures(AudioData audioData, int windowSize, 
            int hopSize, int numBins) throws AudioProcessingException {
        
        // 转换为单声道 / Convert to mono
        AudioData monoAudio = audioData.isMono() ? audioData : AudioProcessor.stereoToMono(audioData);
        IVector<Double> samples = monoAudio.getSamples();
        
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        IMatrix<Double> hpcpMatrix = Linalg.zeros(numBins, numFrames);
        
        // 创建汉宁窗 / Create Hanning window
        IVector<Double> window = createHanningWindow(windowSize);
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            // 提取窗口信号 / Extract windowed signal
            IVector<Double> windowedSignal = extractWindow(samples, start, end, window);
            
            // 计算FFT / Calculate FFT
            Complex[] fftResult = RereFFT.fft(convertToComplex(windowedSignal));
            
            // 计算HPCP特征 / Calculate HPCP features
            IVector<Double> hpcpVector = calculateHPCP(fftResult, audioData.getSampleRate(), numBins);
            
            // 存储到矩阵中 / Store in matrix
            for (int bin = 0; bin < numBins; bin++) {
                hpcpMatrix.set(bin, frame, hpcpVector.get(bin));
            }
        }
        
        return hpcpMatrix;
    }
    
    /**
     * 音调稳定性特征 / Pitch Stability Features
     * <p>
     * 分析音频中音调的稳定性和变化情况，包括音调变化率、音调偏差等。
     * 这些特征对于音乐表现力分析和演奏质量评估很有用。
     * </p>
     * <p>
     * Analyze pitch stability and variation in audio, including pitch change rate, pitch deviation, etc.
     * These features are useful for musical expression analysis and performance quality assessment.
     * </p>
     *
     * @param audioData 输入音频数据 / Input audio data
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return 音调稳定性特征 / Pitch stability features
     * @throws AudioProcessingException 处理过程中发生错误 / Error during processing
     */
    public static IVector<Double> extractPitchStabilityFeatures(AudioData audioData, int windowSize, int hopSize) 
            throws AudioProcessingException {
        
        // 转换为单声道 / Convert to mono
        AudioData monoAudio = audioData.isMono() ? audioData : AudioProcessor.stereoToMono(audioData);
        IVector<Double> samples = monoAudio.getSamples();
        
        int numFrames = (samples.length() - windowSize) / hopSize + 1;
        IVector<Double> pitchTrack = Linalg.zeros(numFrames);
        
        // 提取音调轨迹 / Extract pitch track
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            int end = Math.min(start + windowSize, samples.length());
            
            IVector<Double> frameData = samples.slice(start, end);
            double pitch = estimatePitch(frameData, audioData.getSampleRate());
            pitchTrack.set(frame, pitch);
        }
        
        // 计算稳定性特征 / Calculate stability features
        return calculatePitchStabilityMetrics(pitchTrack);
    }
    
    // ================ 私有辅助方法 / Private Helper Methods ================
    
    /**
     * 创建汉宁窗 / Create Hanning window
     */
    private static IVector<Double> createHanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
    }
    
    /**
     * 提取窗口信号 / Extract windowed signal
     */
    private static IVector<Double> extractWindow(IVector<Double> samples, int start, int end, IVector<Double> window) {
        int length = end - start;
        IVector<Double> windowedSignal = Linalg.zeros(window.length());
        
        for (int i = 0; i < Math.min(length, window.length()); i++) {
            if (start + i < samples.length()) {
                windowedSignal.set(i, samples.get(start + i) * window.get(i));
            }
        }
        
        return windowedSignal;
    }
    
    /**
     * 转换为复数数组 / Convert to complex array
     */
    private static Complex[] convertToComplex(IVector<Double> samples) {
        Complex[] complex = new Complex[samples.length()];
        for (int i = 0; i < samples.length(); i++) {
            complex[i] = new Complex(samples.get(i), 0);
        }
        return complex;
    }
    
    /**
     * 从FFT结果计算色度特征 / Calculate chroma features from FFT result
     */
    private static IVector<Double> calculateChromaFromFFT(Complex[] fftResult, double sampleRate) {
        IVector<Double> chroma = Linalg.zeros(12);
        int numBins = fftResult.length / 2; // 只考虑正频率部分 / Only consider positive frequency part
        
        for (int bin = 1; bin < numBins; bin++) { // 跳过DC分量 / Skip DC component
            double frequency = bin * sampleRate / (2 * numBins);
            double magnitude = fftResult[bin].magnitude();
            
            // 将频率映射到色度 / Map frequency to chroma
            int chromaIndex = frequencyToChroma(frequency);
            chroma.set(chromaIndex, chroma.get(chromaIndex) + magnitude);
        }
        
        // 归一化 / Normalize
        double sum = chroma.sum();
        if (sum > 0) {
            for (int i = 0; i < 12; i++) {
                chroma.set(i, chroma.get(i) / sum);
            }
        }
        
        return chroma;
    }
    
    /**
     * 将频率映射到色度索引 / Map frequency to chroma index
     */
    private static int frequencyToChroma(double frequency) {
        if (frequency <= 0) return 0;
        
        // 将频率转换为相对于A4的半音数 / Convert frequency to semitones relative to A4
        double semitones = 12 * Math.log(frequency / A4_FREQUENCY) / Math.log(2);
        
        // 映射到12个色度类别 / Map to 12 chroma classes
        int chromaIndex = ((int) Math.round(semitones) % 12 + 12) % 12;
        
        // 将A(9)映射到C(0)开始的索引 / Map A(9) to C(0) based indexing
        return (chromaIndex + 3) % 12; // A->C的偏移 / A to C offset
    }
    
    /**
     * 从色度特征计算Tonnetz坐标 / Calculate Tonnetz coordinates from chroma features
     */
    private static IVector<Double> calculateTonnetzFromChroma(IVector<Double> chroma) {
        IVector<Double> tonnetz = Linalg.zeros(6);
        
        // 计算三个基本的Tonnetz维度 / Calculate three basic Tonnetz dimensions
        for (int dim = 0; dim < 3; dim++) {
            double real = 0, imag = 0;
            
            for (int chroma_idx = 0; chroma_idx < 12; chroma_idx++) {
                double weight = chroma.get(chroma_idx);
                double angle = 2 * Math.PI * chroma_idx / 12;
                
                // 根据不同的Tonnetz维度调整角度 / Adjust angle for different Tonnetz dimensions
                if (dim == 1) angle *= 7; // 五度圈 / Circle of fifths
                if (dim == 2) angle *= 3; // 大三度 / Major thirds
                
                real += weight * Math.cos(angle);
                imag += weight * Math.sin(angle);
            }
            
            tonnetz.set(dim * 2, real);
            tonnetz.set(dim * 2 + 1, imag);
        }
        
        return tonnetz;
    }
    
    /**
     * 计算幅度谱 / Calculate magnitude spectrum
     */
    private static IVector<Double> calculateMagnitudeSpectrum(Complex[] fftResult) {
        int numBins = fftResult.length / 2;
        IVector<Double> magnitude = Linalg.zeros(numBins);
        
        for (int i = 0; i < numBins; i++) {
            magnitude.set(i, fftResult[i].magnitude());
        }
        
        return magnitude;
    }
    
    /**
     * 计算频谱对比度 / Calculate spectral contrast
     */
    private static IVector<Double> calculateSpectralContrast(IVector<Double> magnitudeSpectrum, int numBands) {
        IVector<Double> contrast = Linalg.zeros(numBands);
        int binSize = magnitudeSpectrum.length() / numBands;
        
        for (int band = 0; band < numBands; band++) {
            int start = band * binSize;
            int end = Math.min(start + binSize, magnitudeSpectrum.length());
            
            // 计算该频段的峰值和谷值 / Calculate peaks and valleys in this band
            double[] bandData = new double[end - start];
            for (int i = start; i < end; i++) {
                bandData[i - start] = magnitudeSpectrum.get(i);
            }
            
            java.util.Arrays.sort(bandData);
            
            // 取前10%作为峰值，后10%作为谷值 / Take top 10% as peaks, bottom 10% as valleys
            int peakCount = Math.max(1, bandData.length / 10);
            int valleyCount = Math.max(1, bandData.length / 10);
            
            double peakMean = 0, valleyMean = 0;
            
            for (int i = 0; i < peakCount; i++) {
                peakMean += bandData[bandData.length - 1 - i];
            }
            peakMean /= peakCount;
            
            for (int i = 0; i < valleyCount; i++) {
                valleyMean += bandData[i];
            }
            valleyMean /= valleyCount;
            
            // 计算对比度 / Calculate contrast
            double contrastValue = (peakMean > 0 && valleyMean > 0) ? 
                Math.log(peakMean / (valleyMean + 1e-10)) : 0;
            
            contrast.set(band, contrastValue);
        }
        
        return contrast;
    }
    
    /**
     * 计算HPCP特征 / Calculate HPCP features
     */
    private static IVector<Double> calculateHPCP(Complex[] fftResult, double sampleRate, int numBins) {
        IVector<Double> hpcp = Linalg.zeros(numBins);
        int fftSize = fftResult.length;
        
        for (int bin = 1; bin < fftSize / 2; bin++) {
            double frequency = bin * sampleRate / fftSize;
            double magnitude = fftResult[bin].magnitude();
            
            // 计算基频和谐波的贡献 / Calculate contribution of fundamental and harmonics
            for (int harmonic = 1; harmonic <= 5; harmonic++) {
                double harmonicFreq = frequency / harmonic;
                if (harmonicFreq < 80) continue; // 忽略过低的频率 / Ignore too low frequencies
                
                int hpcpIndex = frequencyToHPCPIndex(harmonicFreq, numBins);
                double weight = magnitude / (harmonic * harmonic); // 谐波权重递减 / Harmonic weight decreases
                
                hpcp.set(hpcpIndex, hpcp.get(hpcpIndex) + weight);
            }
        }
        
        // 归一化 / Normalize
        double sum = hpcp.sum();
        if (sum > 0) {
            for (int i = 0; i < numBins; i++) {
                hpcp.set(i, hpcp.get(i) / sum);
            }
        }
        
        return hpcp;
    }
    
    /**
     * 将频率映射到HPCP索引 / Map frequency to HPCP index
     */
    private static int frequencyToHPCPIndex(double frequency, int numBins) {
        double semitones = 12 * Math.log(frequency / A4_FREQUENCY) / Math.log(2);
        int index = (int) Math.round(semitones * numBins / 12) % numBins;
        return (index + numBins) % numBins;
    }
    
    /**
     * 估计音调 / Estimate pitch
     */
    private static double estimatePitch(IVector<Double> frameData, double sampleRate) {
        // 使用自相关方法估计音调 / Use autocorrelation method to estimate pitch
        IVector<Double> autocorr = calculateAutocorrelation(frameData);
        
        int minLag = (int) (sampleRate / 800); // 最高800Hz / Maximum 800Hz
        int maxLag = (int) (sampleRate / 80);  // 最低80Hz / Minimum 80Hz
        
        double maxCorr = 0;
        int maxLag_idx = minLag;
        
        for (int lag = minLag; lag < Math.min(maxLag, autocorr.length()); lag++) {
            if (autocorr.get(lag) > maxCorr) {
                maxCorr = autocorr.get(lag);
                maxLag_idx = lag;
            }
        }
        
        return sampleRate / maxLag_idx;
    }
    
    /**
     * 计算自相关 / Calculate autocorrelation
     */
    private static IVector<Double> calculateAutocorrelation(IVector<Double> signal) {
        int maxLag = signal.length() / 2;
        IVector<Double> autocorr = Linalg.zeros(maxLag);
        
        for (int lag = 0; lag < maxLag; lag++) {
            double sum = 0;
            int count = 0;
            
            for (int i = 0; i < signal.length() - lag; i++) {
                sum += signal.get(i) * signal.get(i + lag);
                count++;
            }
            
            autocorr.set(lag, count > 0 ? sum / count : 0);
        }
        
        return autocorr;
    }
    
    /**
     * 计算音调稳定性指标 / Calculate pitch stability metrics
     */
    private static IVector<Double> calculatePitchStabilityMetrics(IVector<Double> pitchTrack) {
        IVector<Double> metrics = Linalg.zeros(4);
        
        if (pitchTrack.length() < 2) {
            return metrics;
        }
        
        // 1. 音调变化率 / Pitch change rate
        double totalChange = 0;
        int validFrames = 0;
        
        for (int i = 1; i < pitchTrack.length(); i++) {
            double prev = pitchTrack.get(i - 1);
            double curr = pitchTrack.get(i);
            
            if (prev > 0 && curr > 0) {
                totalChange += Math.abs(curr - prev) / prev;
                validFrames++;
            }
        }
        
        double changeRate = validFrames > 0 ? totalChange / validFrames : 0;
        metrics.set(0, changeRate);
        
        // 2. 音调标准差 / Pitch standard deviation
        double mean = 0;
        int count = 0;
        for (int i = 0; i < pitchTrack.length(); i++) {
            if (pitchTrack.get(i) > 0) {
                mean += pitchTrack.get(i);
                count++;
            }
        }
        mean = count > 0 ? mean / count : 0;
        
        double variance = 0;
        for (int i = 0; i < pitchTrack.length(); i++) {
            if (pitchTrack.get(i) > 0) {
                double diff = pitchTrack.get(i) - mean;
                variance += diff * diff;
            }
        }
        double stdDev = count > 1 ? Math.sqrt(variance / (count - 1)) : 0;
        metrics.set(1, stdDev);
        
        // 3. 有声帧比例 / Voiced frame ratio
        double voicedRatio = count > 0 ? (double) count / pitchTrack.length() : 0;
        metrics.set(2, voicedRatio);
        
        // 4. 音调平均值 / Pitch mean
        metrics.set(3, mean);
        
        return metrics;
    }
}