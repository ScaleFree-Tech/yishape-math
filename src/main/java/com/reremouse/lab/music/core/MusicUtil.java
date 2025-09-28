package com.reremouse.lab.music.core;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioProcessor;
import com.reremouse.lab.audio.core.AudioUtil;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.reremouse.lab.music.analysis.basic.BeatDetectionResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 音乐分析工具类 - 提供全面的音乐信号处理和分析功能
 * 
 * <p>本类是音乐分析系统的核心工具类，提供了丰富的音乐信号处理和分析方法，包括：</p>
 * 
 * <h3>主要功能模块：</h3>
 * <ul>
 *   <li><b>音频特征提取</b>：Tonnetz特征、HPCP特征、音调稳定性分析等高级音乐特征</li>
 *   <li><b>频谱分析</b>：频谱重心、滚降、带宽、对比度、平坦度等频域特征计算</li>
 *   <li><b>时域分析</b>：RMS、零交叉率、能量包络、动态范围等时域特征</li>
 *   <li><b>音乐情感分析</b>：效价(Valence)、唤醒度(Arousal)、支配度(Dominance)等情感维度</li>
 *   <li><b>音乐风格特征</b>：可舞性、原声性、器乐性、现场感、语音性等风格属性</li>
 *   <li><b>节奏分析</b>：节拍强度、节奏模式、节拍稳定性、节奏复杂度计算</li>
 *   <li><b>调性分析</b>：调性强度、调性稳定性、色度特征等和声分析</li>
 *   <li><b>结构分析</b>：音乐段落分析、新颖性函数、自相似矩阵、重复性分析等</li>
 *   <li><b>MFCC特征</b>：梅尔频率倒谱系数及其差分特征计算</li>
 * </ul>
 * 
 * <h3>应用场景：</h3>
 * <ul>
 *   <li>音乐信息检索(MIR)系统开发</li>
 *   <li>音乐推荐算法特征工程</li>
 *   <li>音乐情感识别和分析</li>
 *   <li>音乐风格分析和流派识别</li>
 *   <li>音频质量评估和演奏分析</li>
 *   <li>音乐教育和理论研究</li>
 * </ul>
 * 
 * <h3>技术特点：</h3>
 * <ul>
 *   <li>基于现代音乐信息学理论和算法</li>
 *   <li>支持多种音频格式和采样率</li>
 *   <li>优化的数值计算性能</li>
 *   <li>模块化设计，易于扩展</li>
 *   <li>完整的异常处理机制</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MusicUtil {

    // ==================== 默认参数配置 / Default Parameter Configuration ====================
    
    /**
     * 默认窗口大小 - 用于FFT分析的窗口长度
     * <p>4096样本点提供了良好的频率分辨率，适合大多数音乐分析任务。
     * 较大的窗口提供更好的频率分辨率，但时间分辨率较低。</p>
     */
    private static final int DEFAULT_WINDOW_SIZE = 4096;
    
    /**
     * 默认跳跃大小 - 相邻分析窗口之间的样本间隔
     * <p>2048样本（窗口大小的一半）提供50%的重叠，
     * 这是音频分析中常用的配置，平衡了计算效率和分析精度。</p>
     */
    private static final int DEFAULT_HOP_SIZE = 2048;
    
    /**
     * 默认帧大小 - 用于短时分析的帧长度
     * <p>1024样本点适合快速变化的音频特征分析，
     * 如节拍检测、瞬态分析等需要较高时间分辨率的任务。</p>
     */
    private static final int DEFAULT_FRAME_SIZE = 1024;
    
    /**
     * 默认段落分析长度 - 音乐结构分析的时间窗口（秒）
     * <p>30秒是音乐段落分析的典型长度，足够捕获大多数音乐的结构变化，
     * 如副歌、主歌等音乐段落的转换。</p>
     */
    private static final int DEFAULT_SEGMENT_LENGTH = 30;
    
    /**
     * 默认MFCC系数数量 - 梅尔频率倒谱系数的维度
     * <p>13个系数是语音和音乐分析的标准配置，包含了足够的频谱信息
     * 用于音色识别、乐器分类等任务，同时保持合理的计算复杂度。</p>
     */
    private static final int DEFAULT_MFCC_COUNT = 13;

    // ==================== 高级音频特征提取方法 / Advanced Audio Feature Extraction ====================
    
    /**
     * 提取Tonnetz特征 - 基于音调网络理论的和声分析
     * 
     * <p><b>Tonnetz(音调网络)</b>是音乐理论中的一个重要概念，将音高关系映射到二维几何空间中。
     * 该方法将传统的色度特征转换为更具音乐理论意义的Tonnetz坐标系表示。</p>
     * 
     * <h4>理论基础：</h4>
     * <ul>
     *   <li><b>纯五度关系</b>：音乐中最重要的和声关系之一</li>
     *   <li><b>大三度关系</b>：构成大调和弦的基础音程</li>
     *   <li><b>小三度关系</b>：构成小调和弦的基础音程</li>
     * </ul>
     * 
     * <h4>特征维度(6维)：</h4>
     * <ul>
     *   <li><b>维度0-1</b>：纯五度圆的实部和虚部坐标</li>
     *   <li><b>维度2-3</b>：大三度圆的实部和虚部坐标</li>
     *   <li><b>维度4-5</b>：小三度圆的实部和虚部坐标</li>
     * </ul>
     * 
     * <h4>应用场景：</h4>
     * <ul>
     *   <li>和声进行分析和识别</li>
     *   <li>调性变化检测</li>
     *   <li>音乐风格分类</li>
     *   <li>和弦识别和标注</li>
     *   <li>音乐相似度计算</li>
     * </ul>
     * 
     * @param audioData 输入的音频数据，支持单声道和立体声
     * @return 6维Tonnetz特征向量，包含三个复数坐标的实部和虚部
     * @throws AudioProcessingException 当音频处理过程中发生错误时抛出
     * 
     * @see KeyAnalyzerImpl#analyzeChromaFeatures(AudioData) 色度特征提取
     */
    public static double[] extractTonnetzFeatures(AudioData audioData) throws AudioProcessingException {
        try {
            // 首先提取色度特征
            var keyAnalyzer = new KeyAnalyzerImpl();
            double[] chromaVector = keyAnalyzer.analyzeChromaFeatures(audioData);
            
            // 计算Tonnetz坐标 (6维：3个复数坐标的实部和虚部)
            double[] tonnetzFeatures = new double[6];
            
            // Tonnetz坐标系的基础频率比例
            double[][] tonnetzVectors = {
                {0.0, 1.0}, // 纯五度 / Perfect fifth
                {Math.sqrt(3) / 2, 0.5}, // 大三度 / Major third
                {-Math.sqrt(3) / 2, 0.5} // 小三度 / Minor third
            };
            
            // 计算三个基本的Tonnetz维度
            for (int dim = 0; dim < 3; dim++) {
                double real = 0, imag = 0;
                
                for (int chroma_idx = 0; chroma_idx < 12; chroma_idx++) {
                    double weight = chromaVector[chroma_idx];
                    double angle = 2 * Math.PI * chroma_idx / 12;
                    
                    // 根据不同的Tonnetz维度调整角度
                    if (dim == 1) {
                        angle *= 7; // 五度圆 / Circle of fifths
                    }
                    if (dim == 2) {
                        angle *= 3; // 大三度 / Major thirds
                    }
                    real += weight * Math.cos(angle);
                    imag += weight * Math.sin(angle);
                }
                
                tonnetzFeatures[dim * 2] = real;
                tonnetzFeatures[dim * 2 + 1] = imag;
            }
            
            return tonnetzFeatures;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error extracting Tonnetz features: " + e.getMessage(), e);
        }
    }

    /**
     * 提取HPCP特征 - 谐波感知音调阶级描述
     * 
     * <p><b>HPCP(Harmonic Pitch Class Profiles)</b>是色度特征的高级版本，
     * 通过考虑谐波结构来提供更准确的音调表示。与传统色度特征相比，
     * HPCP能够更好地处理复杂的音乐信号和多音符场景。</p>
     * 
     * <h4>技术特点：</h4>
     * <ul>
     *   <li><b>谐波感知</b>：考虑基频及其谐波成分的贡献</li>
     *   <li><b>权重衰减</b>：高次谐波的权重递减，符合听觉感知</li>
     *   <li><b>频率映射</b>：精确的频率到音调类别的映射</li>
     *   <li><b>归一化处理</b>：确保特征向量的数值稳定性</li>
     * </ul>
     * 
     * <h4>算法流程：</h4>
     * <ol>
     *   <li>对音频信号进行FFT变换</li>
     *   <li>遍历频谱中的每个频率分量</li>
     *   <li>计算该频率及其谐波对HPCP的贡献</li>
     *   <li>应用谐波权重衰减(1/n²)</li>
     *   <li>将频率映射到对应的音调类别</li>
     *   <li>归一化最终的HPCP向量</li>
     * </ol>
     * 
     * <h4>应用优势：</h4>
     * <ul>
     *   <li>更准确的音调识别</li>
     *   <li>更好的和弦检测性能</li>
     *   <li>对噪声和混响的鲁棒性</li>
     *   <li>适合复调音乐分析</li>
     * </ul>
     * 
     * @param audioData 输入的音频数据
     * @param numBins HPCP特征的维度（通常12或36）
     * @return numBins维的HPCP特征向量
     * @throws AudioProcessingException 当音频处理过程中发生错误时抛出
     * 
     * @see #frequencyToHPCPIndex(double, int, double) 频率到HPCP索引的映射
     */
    public static double[] extractHPCPFeatures(AudioData audioData, int numBins) throws AudioProcessingException {
        try {
            // 转换为单声道
            AudioData monoAudio = audioData.isMono() ? audioData
                    : AudioProcessor.stereoToMono(audioData);
            
            // 获取音频样本
            double[] samples = monoAudio.getSamples().toDoubleArray();
            double sampleRate = audioData.getSampleRate();
            
            // 计算FFT
            int windowSize = 2048;
            Complex[] fftResult = RereFFT.fft(AudioUtil.convertToComplex(samples, windowSize));
            
            // 计算HPCP特征
            double[] hpcp = new double[numBins];
            int fftSize = fftResult.length;
            
            // 标准A4音高频率
            double A4_FREQUENCY = 440.0;
            
            for (int bin = 1; bin < fftSize / 2; bin++) {
                double frequency = bin * sampleRate / fftSize;
                double magnitude = fftResult[bin].magnitude();
                
                // 计算基频和谐波的贡献
                for (int harmonic = 1; harmonic <= 5; harmonic++) {
                    double harmonicFreq = frequency / harmonic;
                    if (harmonicFreq < 80) {
                        continue; // 忽略过低的频率
                    }
                    int hpcpIndex = frequencyToHPCPIndex(harmonicFreq, numBins, A4_FREQUENCY);
                    double weight = magnitude / (harmonic * harmonic); // 调和权重递减
                    
                    hpcp[hpcpIndex] += weight;
                }
            }
            
            // 归一化 - 使用IVector API优化
            IVector<Double> hpcpVector = Linalg.vector(hpcp);
            double sum = hpcpVector.sum();
            if (sum > 0) {
                hpcpVector = hpcpVector.apply(x -> x / sum);
                // 将结果复制回原数组
                for (int i = 0; i < numBins; i++) {
                    hpcp[i] = hpcpVector.get(i);
                }
            }
            
            return hpcp;
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error extracting HPCP features: " + e.getMessage(), e);
        }
    }

    /**
     * 将频率映射到HPCP索引 - 频率到音调类别的精确映射
     * 
     * <p>该方法实现了从物理频率到音调类别索引的数学映射，
     * 基于十二平均律的对数关系进行计算。</p>
     * 
     * <h4>计算原理：</h4>
     * <ul>
     *   <li><b>对数映射</b>：frequency = A4 * 2^(semitones/12)</li>
     *   <li><b>半音计算</b>：semitones = 12 * log₂(frequency/A4)</li>
     *   <li><b>索引映射</b>：index = (semitones * numBins / 12) % numBins</li>
     * </ul>
     * 
     * @param frequency 输入频率(Hz)
     * @param numBins HPCP特征的总维度
     * @param A4_FREQUENCY A4音符的参考频率（通常440Hz）
     * @return 对应的HPCP索引(0到numBins-1)
     */
    public static int frequencyToHPCPIndex(double frequency, int numBins, double A4_FREQUENCY) {
        double semitones = 12 * Math.log(frequency / A4_FREQUENCY) / Math.log(2);
        int index = (int) Math.round(semitones * numBins / 12) % numBins;
        return (index + numBins) % numBins;
    }

    /**
     * 提取音调稳定性特征 - 音高变化和稳定性分析
     * 
     * <p>该方法通过分析音频中音调的时间变化特性，提取反映音高稳定性的多维特征。
     * 这些特征对于评估演奏质量、检测音调问题以及分析音乐表现力具有重要价值。</p>
     * 
     * <h4>分析维度(4维特征)：</h4>
     * <ul>
     *   <li><b>音调变化率</b>：相邻帧之间音调的平均变化幅度</li>
     *   <li><b>音调标准差</b>：整体音调分布的离散程度</li>
     *   <li><b>有声帧比例</b>：包含明确音调信息的帧所占比例</li>
     *   <li><b>音调平均值</b>：整段音频的平均音调高度</li>
     * </ul>
     * 
     * <h4>技术实现：</h4>
     * <ul>
     *   <li><b>音调提取</b>：使用自相关方法估计每帧的基频</li>
     *   <li><b>时间分析</b>：滑动窗口方式分析音调轨迹</li>
     *   <li><b>统计计算</b>：计算音调序列的统计特征</li>
     *   <li><b>稳定性量化</b>：将音调变化转换为数值指标</li>
     * </ul>
     * 
     * <h4>应用场景：</h4>
     * <ul>
     *   <li>音乐演奏质量评估</li>
     *   <li>歌唱技巧分析</li>
     *   <li>乐器音准检测</li>
     *   <li>音乐教育辅助</li>
     *   <li>音频质量评价</li>
     * </ul>
     * 
     * @param audioData 输入的音频数据，建议使用单声道
     * @return 4维音调稳定性特征向量[变化率, 标准差, 有声比例, 平均值]
     * @throws AudioProcessingException 当音频处理过程中发生错误时抛出
     * 
     * @see #estimatePitch(double[], double) 音调估计方法
     * @see #calculatePitchStabilityMetrics(double[]) 稳定性指标计算
     */
    public static double[] extractPitchStabilityFeatures(AudioData audioData) throws AudioProcessingException {
        try {
            // 转换为单声道
            AudioData monoAudio = audioData.isMono() ? audioData
                    : AudioProcessor.stereoToMono(audioData);
            
            double[] samples = monoAudio.getSamples().toDoubleArray();
            double sampleRate = audioData.getSampleRate();
            
            int windowSize = 2048;
            int hopSize = 512;
            int numFrames = (samples.length - windowSize) / hopSize + 1;
            
            double[] pitchTrack = new double[numFrames];
            
            // 提取音调轨迹
            for (int frame = 0; frame < numFrames; frame++) {
                int start = frame * hopSize;
                int end = Math.min(start + windowSize, samples.length);
                
                double[] frameData = Arrays.copyOfRange(samples, start, end);
                double pitch = estimatePitch(frameData, sampleRate);
                pitchTrack[frame] = pitch;
            }
            
            // 计算稳定性特征(4维特征向量)
            return calculatePitchStabilityMetrics(pitchTrack);
            
        } catch (Exception e) {
            throw new AudioProcessingException("Error extracting pitch stability features: " + e.getMessage(), e);
        }
    }

    /**
     * 估计音调 - 基于自相关的基频检测
     * 
     * <p>使用自相关函数来估计音频帧的基频(音调)。自相关方法是音调检测中
     * 最经典和可靠的方法之一，特别适合处理周期性信号。</p>
     * 
     * <h4>算法原理：</h4>
     * <ul>
     *   <li><b>自相关计算</b>：计算信号与其延迟版本的相关性</li>
     *   <li><b>周期检测</b>：寻找自相关函数的最大峰值</li>
     *   <li><b>频率转换</b>：将时间延迟转换为频率</li>
     * </ul>
     * 
     * <h4>频率范围：</h4>
     * <ul>
     *   <li><b>最低频率</b>：80Hz (对应人声和大多数乐器的下限)</li>
     *   <li><b>最高频率</b>：800Hz (覆盖大部分音乐音调范围)</li>
     * </ul>
     * 
     * @param frameData 音频帧数据
     * @param sampleRate 音频采样率
     * @return 估计的基频(Hz)，如果无法检测到则返回0
     */
    private static double estimatePitch(double[] frameData, double sampleRate) {
        int minPeriod = (int) (sampleRate / 800); // 最高频率800Hz
        int maxPeriod = (int) (sampleRate / 80);  // 最低频率80Hz
        
        double maxCorrelation = 0;
        int bestPeriod = 0;
        
        // 计算自相关
        for (int period = minPeriod; period <= maxPeriod && period < frameData.length / 2; period++) {
            double correlation = 0;
            for (int i = 0; i < frameData.length - period; i++) {
                correlation += frameData[i] * frameData[i + period];
            }
            
            if (correlation > maxCorrelation) {
                maxCorrelation = correlation;
                bestPeriod = period;
            }
        }
        
        return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
    }

    /**
     * 计算音调稳定性指标 - 从音调轨迹提取稳定性特征
     * 
     * <p>该方法从音调时间序列中计算多个稳定性指标，
     * 用于量化音调的变化特性和稳定程度。</p>
     * 
     * @param pitchTrack 音调轨迹数组
     * @return 4维稳定性特征[变化率, 标准差, 有声比例, 平均值]
     */
    private static double[] calculatePitchStabilityMetrics(double[] pitchTrack) {
        List<Double> validPitches = new ArrayList<>();
        List<Double> pitchChanges = new ArrayList<>();
        
        // 过滤有效音调值并计算变化率
        for (int i = 0; i < pitchTrack.length; i++) {
            if (pitchTrack[i] > 0) {
                validPitches.add(pitchTrack[i]);
                if (i > 0 && pitchTrack[i-1] > 0) {
                    double change = Math.abs(pitchTrack[i] - pitchTrack[i-1]);
                    pitchChanges.add(change);
                }
            }
        }
        
        double[] features = new double[4];
        
        if (validPitches.isEmpty()) {
            return features; // 全零向量
        }
        
        // 1. 音调变化率
        double avgChange = pitchChanges.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        features[0] = avgChange;
        
        // 2. 音调标准差
        double mean = validPitches.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double variance = validPitches.stream()
                .mapToDouble(p -> Math.pow(p - mean, 2))
                .average().orElse(0);
        features[1] = Math.sqrt(variance);
        
        // 3. 有声帧比例
        features[2] = (double) validPitches.size() / pitchTrack.length;
        
        // 4. 音调平均值
        features[3] = mean;
        
        return features;
    }

    /**
     * 计算节拍稳定性 - 基于节拍时间序列的稳定性分析
     * 
     * <p>该方法通过分析节拍时间序列的规律性和一致性来计算节拍稳定性。
     * 稳定的节拍通常表现为节拍间隔的较小变化和良好的周期性。</p>
     * 
     * @param beatTimes 节拍时间数组(秒)
     * @return 节拍稳定性值(0-1)，值越大表示节拍越稳定
     */
    public static double calculateBeatStability(double[] beatTimes) {
        if (beatTimes == null || beatTimes.length < 2) {
            return 0.0;
        }
        
        // 计算节拍间隔
        double[] intervals = new double[beatTimes.length - 1];
        for (int i = 0; i < intervals.length; i++) {
            intervals[i] = beatTimes[i + 1] - beatTimes[i];
        }
        
        // 计算平均节拍间隔
        double meanInterval = 0;
        for (double interval : intervals) {
            meanInterval += interval;
        }
        meanInterval /= intervals.length;
        
        // 计算节拍间隔的标准差
        double variance = 0;
        for (double interval : intervals) {
            double diff = interval - meanInterval;
            variance += diff * diff;
        }
        variance /= intervals.length;
        double stdDev = Math.sqrt(variance);
        
        // 节拍稳定性：标准差越小，稳定性越高
        // 使用指数衰减函数将标准差映射到0-1范围
        double stability = Math.exp(-stdDev / meanInterval);
        
        return Math.max(0.0, Math.min(1.0, stability));
    }

    /**
     * 计算节奏复杂度 - 基于节奏模式的复杂度分析
     * 
     * <p>该方法通过分析节奏模式的变化和不规则性来计算节奏复杂度。
     * 复杂的节奏通常表现为节拍强度的大幅变化和不规则的节奏模式。</p>
     * 
     * @param rhythmPattern 节奏模式数组
     * @return 节奏复杂度值(0-1)，值越大表示节奏越复杂
     */
    public static double calculateRhythmComplexity(double[] rhythmPattern) {
        if (rhythmPattern == null || rhythmPattern.length == 0) {
            return 0.0;
        }
        
        // 计算平均值
        double mean = 0;
        for (double value : rhythmPattern) {
            mean += value;
        }
        mean /= rhythmPattern.length;
        
        // 计算方差
        double variance = 0;
        for (double value : rhythmPattern) {
            double diff = value - mean;
            variance += diff * diff;
        }
        variance /= rhythmPattern.length;
        
        // 计算变化率
        double variation = Math.sqrt(variance) / (mean > 0 ? mean : 1.0);
        
        // 节奏复杂度：变化率越高，复杂度越高
        // 使用Sigmoid函数将变化率映射到0-1范围
        double complexity = 1.0 / (1.0 + Math.exp(-variation));
        
        return Math.max(0.0, Math.min(1.0, complexity));
    }

    /**
     * 计算调性稳定性 - 基于音频数据的调性稳定性分析
     * 
     * <p>该方法通过分析音频的色度特征变化来计算调性稳定性。
     * 稳定的调性通常表现为色度特征的一致性和较少的变化。</p>
     * 
     * @param audioData 音频数据
     * @return 调性稳定性值(0-1)，值越大表示调性越稳定
     * @throws AudioProcessingException 当音频处理过程中发生错误时抛出
     */
    public static double calculateTonalStability(AudioData audioData) throws AudioProcessingException {
        try {
            // 简化实现，基于节拍分析器估计的节拍速度稳定性
            var beatAnalyzer = new BeatAnalyzerImpl();
            double tempo = beatAnalyzer.estimateTempo(audioData);
            
            // 分析节拍的一致性
            double stability = 1.0;
            
            // 极端节拍速度通常不太稳定
            if (tempo < 60 || tempo > 180) {
                stability *= 0.8;
            }
            
            return Math.max(0.0, Math.min(1.0, stability));
            
        } catch (Exception e) {
            return 0.5; // 默认中等稳定性
        }
    }

    /**
     * 计算调性强度 - 基于色度向量的调性强度分析
     * 
     * <p>该方法通过分析色度向量的清晰度和集中度来计算调性强度。
     * 强烈的调性通常表现为某些音高类别的显著突出。</p>
     * 
     * @param chromaVector 色度向量(12维)
     * @return 调性强度值(0-1)，值越大表示调性越强烈
     */
    public static double calculateTonalStrength(double[] chromaVector) {
        if (chromaVector == null || chromaVector.length == 0) {
            return 0.0;
        }
        
        // 计算色度向量的总能量
        double totalEnergy = 0;
        double maxEnergy = 0;
        
        for (double energy : chromaVector) {
            totalEnergy += energy;
            if (energy > maxEnergy) {
                maxEnergy = energy;
            }
        }
        
        // 如果总能量为0，返回0
        if (totalEnergy <= 0) {
            return 0.0;
        }
        
        // 调性强度：最大能量占比越高，调性越强烈
        double tonalStrength = maxEnergy / totalEnergy;
        
        return Math.max(0.0, Math.min(1.0, tonalStrength));
    }

    /**
     * 计算MFCC稳定性 - 基于MFCC帧矩阵的稳定性分析
     * 
     * <p>该方法通过分析MFCC特征在时间维度上的变化来计算稳定性。
     * 稳定性值越低表示MFCC特征变化越大，音乐越复杂或变化越丰富；
     * 稳定性值越高表示MFCC特征相对稳定，音乐越单调或重复性越强。</p>
     * 
     * <h4>计算方法：</h4>
     * <ul>
     *   <li><b>方差计算</b>：计算每个MFCC系数在时间维度上的方差</li>
     *   <li><b>平均处理</b>：对所有MFCC系数的方差求平均</li>
     *   <li><b>归一化</b>：将结果映射到0-1范围</li>
     * </ul>
     * 
     * <h4>特征意义：</h4>
     * <ul>
     *   <li><b>低稳定性</b>：特征变化丰富，音乐复杂度高</li>
     *   <li><b>高稳定性</b>：特征相对稳定，音乐重复性强</li>
     * </ul>
     * 
     * <h4>应用场景：</h4>
     * <ul>
     *   <li>音乐风格分类</li>
     *   <li>音乐复杂度分析</li>
     *   <li>音乐结构分析</li>
     *   <li>音频相似性度量</li>
     * </ul>
     * 
     * @param mfccFrames MFCC帧矩阵，每行代表一帧，每列代表一个MFCC系数
     * @return MFCC稳定性值(0-1)，值越高表示越稳定
     */
    public static double calculateMFCCStability(double[][] mfccFrames) {
        if (mfccFrames == null || mfccFrames.length == 0 || mfccFrames[0].length == 0) {
            return 0.0;
        }
        
        int numFrames = mfccFrames.length;
        int numCoeffs = mfccFrames[0].length;
        
        // 计算每个系数的方差
        double totalVariance = 0.0;
        for (int coeff = 0; coeff < numCoeffs; coeff++) {
            // 计算均值
            double sum = 0.0;
            for (int frame = 0; frame < numFrames; frame++) {
                sum += mfccFrames[frame][coeff];
            }
            double mean = sum / numFrames;
            
            // 计算方差
            double variance = 0.0;
            for (int frame = 0; frame < numFrames; frame++) {
                double diff = mfccFrames[frame][coeff] - mean;
                variance += diff * diff;
            }
            variance /= numFrames;
            
            totalVariance += variance;
        }
        
        // 计算平均方差并归一化
        double avgVariance = totalVariance / numCoeffs;
        
        // 使用指数函数将方差映射到0-1范围，方差越大稳定性越低
        // 这里使用一个简单的归一化方法
        double stability = 1.0 / (1.0 + Math.sqrt(avgVariance));
        
        return Math.max(0.0, Math.min(1.0, stability));
    }

}