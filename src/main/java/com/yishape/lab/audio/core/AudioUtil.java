package com.yishape.lab.audio.core;

import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.SignalUtilities;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;

import java.util.Map;

/**
 *
 * @author lteb2
 */
public class AudioUtil {

    /**
     * 默认窗口大小 - 用于FFT分析的窗口长度
     * <p>
     * 4096样本点提供了良好的频率分辨率，适合大多数音乐分析任务。 较大的窗口提供更好的频率分辨率，但时间分辨率较低。</p>
     */
    private static final int DEFAULT_WINDOW_SIZE = 4096;

    /**
     * 默认跳跃大小 - 相邻分析窗口之间的样本间隔
     * <p>
     * 2048样本点(窗口大小的一半)提供了50%的重叠， 这是音频分析中常用的配置，平衡了计算效率和分析精度。</p>
     */
    private static final int DEFAULT_HOP_SIZE = 2048;

    /**
     * 默认帧大小 - 用于短时分析的帧长度
     * <p>
     * 1024样本点适合快速变化的音频特征分析， 如节拍检测、瞬态分析等需要较高时间分辨率的任务。</p>
     */
    private static final int DEFAULT_FRAME_SIZE = 1024;

    /**
     * 倒梅尔系数的默认个数
     */
    private static final int DEFAULT_MFCC_COUNT = 13;
    
    
    /**
     * 执行完整的FFT处理流程
     *
     * <p>
     * 该方法封装了音频FFT处理的标准流程：
     * <ol>
     * <li>应用Hamming窗函数</li>
     * <li>将实数信号转换为复数数组</li>
     * <li>执行快速傅里叶变换</li>
     * </ol>
     * </p>
     *
     * @param audioData 音频数据
     * @param windowSize 窗口大小
     * @return FFT结果的复数数组
     * @throws AudioProcessingException 当音频处理过程中发生错误时抛出
     */
    public static Complex[] processFFT(AudioData audioData, int windowSize) throws AudioProcessingException {
        try {
            // 应用窗函数
            IVector<Double> windowed = applyWindow(audioData.getSamples(), windowSize);

            // 转换为复数数组
            Complex[] input = convertToComplex(windowed);

            // 零填充确保长度为2的幂
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);

            // 执行FFT
            return RereFFT.fft(paddedInput);
        } catch (Exception e) {
            throw new AudioProcessingException("Error processing FFT: " + e.getMessage(), e);
        }
    }

    /**
     * 执行完整的FFT处理流程（使用默认窗口大小）
     *
     * @param audioData 音频数据
     * @return FFT结果的复数数组
     * @throws AudioProcessingException 当音频处理过程中发生错误时抛出
     */
    public static Complex[] processFFT(AudioData audioData) throws AudioProcessingException {
        return processFFT(audioData, DEFAULT_WINDOW_SIZE);
    }

    /**
     * 执行FFT处理流程（从已加窗的IVector数据）
     *
     * <p>
     * 该方法直接对已加窗的音频数据执行FFT变换，适用于需要对音频帧进行 分帧处理的场景。</p>
     *
     * @param windowed 已加窗的音频数据向量
     * @return FFT结果的复数数组
     */
    public static Complex[] processFFT(IVector<Double> windowed) {
        // 转换为复数数组
        Complex[] input = convertToComplex(windowed);

        // 零填充确保长度为2的幂
        Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);

        // 执行FFT
        return RereFFT.fft(paddedInput);
    }

    /**
     * 转换为复数数组 - 实数数组到复数数组的转换
     *
     * <p>
     * 将实数音频样本转换为复数数组，虚部设为0。这是进行FFT等频域分析 的必要预处理步骤。</p>
     *
     * @param samples 实数音频样本向量
     * @return 对应的复数数组，虚部为0
     */
    public static Complex[] convertToComplex(IVector<Double> samples) {
        return convertToComplex(samples.toDoubleArray());
    }

    /**
     * 转换为复数数组 - 实数数组到复数数组的转换
     *
     * @param samples 实数音频样本数组
     * @return 对应的复数数组，虚部为0
     */
    public static Complex[] convertToComplex(double[] samples) {
        Complex[] complex = new Complex[samples.length];
        for (int i = 0; i < samples.length; i++) {
            complex[i] = new Complex(samples[i], 0.0);
        }
        return complex;
    }

    /**
     * 转换为复数数组 - 实数数组到复数数组的转换
     *
     * <p>
     * 将实数音频样本转换为复数数组，虚部设为0。这是进行FFT等频域分析 的必要预处理步骤，同时支持窗口大小限制。</p>
     *
     * <h4>转换规则：</h4>
     * <ul>
     * <li><b>实部</b>：保持原始音频样本值</li>
     * <li><b>虚部</b>：设置为0</li>
     * <li><b>长度限制</b>：取样本长度和窗口大小的最小值</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>FFT变换前的数据预处理</li>
     * <li>频域分析算法的输入准备</li>
     * <li>分帧处理中的数据格式转换</li>
     * </ul>
     *
     * @param samples 实数音频样本数组
     * @param windowSize 窗口大小限制
     * @return 对应的复数数组，虚部为0
     */
    public static Complex[] convertToComplex(double[] samples, int windowSize) {
        int length = Math.min(samples.length, windowSize);
        Complex[] complex = new Complex[length];
        for (int i = 0; i < length; i++) {
            complex[i] = new Complex(samples[i], 0);
        }
        return complex;
    }

    /**
     * 增强的频谱对比度特征 / Enhanced Spectral Contrast Features
     * <p>
     * 计算多个频段的频谱对比度，捕获音频的谐波和非谐波成分的相对强度。 频谱对比度特征对音乐的音色和质感有很好的描述能力。
     * </p>
     * <p>
     * Calculate spectral contrast in multiple frequency bands, capturing
     * relative intensity of harmonic and non-harmonic components. Spectral
     * contrast features have good descriptive ability for timbre and texture of
     * music.
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
            Complex[] input = convertToComplex(windowedSignal);
            // 零填充确保长度为2的幂
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);
            Complex[] fftResult = RereFFT.fft(paddedInput);

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
     * 创建汉宁窗 / Create Hanning window
     * 
     * <p>汉宁窗（Hanning Window）是一种常用的窗函数，用于减少FFT分析中的频谱泄漏。
     * 汉宁窗是余弦窗的一种，具有良好的频率选择性和较低的旁瓣水平。</p>
     * 
     * <h4>数学公式：</h4>
     * <p>w(n) = 0.5 * (1 - cos(2π * n / (N-1)))</p>
     * <p>其中 n ∈ [0, N-1]，N 是窗口长度</p>
     * 
     * <h4>特点：</h4>
     * <ul>
     * <li><b>平滑过渡</b>：窗口两端平滑衰减到0，中间为1</li>
     * <li><b>减少泄漏</b>：有效降低FFT频谱泄漏</li>
     * <li><b>频率分辨率</b>：相比矩形窗，频率分辨率略低但旁瓣更小</li>
     * </ul>
     * 
     * @param size 窗口大小
     * @return 汉宁窗向量
     */
    public static IVector<Double> createHanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            // 汉宁窗公式：w(n) = 0.5 * (1 - cos(2π * n / (N-1)))
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
    }

    /**
     * 提取窗口信号 / Extract windowed signal
     * 
     * <p>从音频样本中提取指定范围的数据，并应用窗函数。
     * 如果提取的样本数少于窗口长度，剩余部分将填充为0。</p>
     * 
     * @param samples 原始音频样本向量
     * @param start 起始索引
     * @param end 结束索引（不包含）
     * @param window 窗函数向量
     * @return 加窗后的信号向量，长度等于window.length()
     */
    public static IVector<Double> extractWindow(IVector<Double> samples, int start, int end, IVector<Double> window) {
        int length = end - start;
        IVector<Double> windowedSignal = Linalg.zeros(window.length());

        // 提取样本并应用窗函数
        for (int i = 0; i < Math.min(length, window.length()); i++) {
            if (start + i < samples.length()) {
                // 样本值乘以对应的窗函数值
                windowedSignal.set(i, samples.get(start + i) * window.get(i));
            }
            // 如果超出样本范围，保持为0（零填充）
        }

        return windowedSignal;
    }

    /**
     * 计算幅度谱 / Calculate magnitude spectrum
     * 
     * <p>从FFT结果中提取幅度谱。由于FFT结果具有共轭对称性（对于实数输入），
     * 只需要取前半部分（0到Nyquist频率）。</p>
     * 
     * <h4>说明：</h4>
     * <ul>
     * <li><b>FFT对称性</b>：实数信号的FFT结果关于Nyquist频率对称</li>
     * <li><b>有效范围</b>：只取前N/2个频率分量（0到fs/2）</li>
     * <li><b>幅度计算</b>：|X(k)| = sqrt(real² + imag²)</li>
     * </ul>
     * 
     * @param fftResult FFT结果复数数组
     * @return 幅度谱向量，长度为FFT长度的一半
     */
    public static IVector<Double> calculateMagnitudeSpectrum(Complex[] fftResult) {
        // 只取前半部分（由于共轭对称性）
        int numBins = fftResult.length / 2;
        IVector<Double> magnitude = Linalg.zeros(numBins);

        for (int i = 0; i < numBins; i++) {
            // 计算复数的模：sqrt(real^2 + imag^2)
            magnitude.set(i, fftResult[i].magnitude());
        }

        return magnitude;
    }

    /**
     * 计算频谱对比度 / Calculate spectral contrast
     * 
     * <p>将频谱分为多个频段，计算每个频段内峰值和谷值的对比度。
     * 频谱对比度特征能够捕捉音频的谐波和非谐波成分的相对强度。</p>
     * 
     * <h4>算法步骤：</h4>
     * <ol>
     * <li>将频谱等分为numBands个频段</li>
     * <li>对每个频段内的幅度值排序</li>
     * <li>取前10%作为峰值，后10%作为谷值</li>
     * <li>计算对比度：log(峰值均值 / 谷值均值)</li>
     * </ol>
     * 
     * @param magnitudeSpectrum 幅度谱向量
     * @param numBands 频段数量
     * @return 每个频段的对比度向量
     */
    public static IVector<Double> calculateSpectralContrast(IVector<Double> magnitudeSpectrum, int numBands) {
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
            double contrastValue = (peakMean > 0 && valleyMean > 0)
                    ? Math.log(peakMean / (valleyMean + 1e-10)) : 0;

            contrast.set(band, contrastValue);
        }

        return contrast;
    }

    /**
     * 计算自相关 - 信号的自相关函数分析
     *
     * <p>
     * 计算信号的自相关函数，用于分析信号的周期性特征。自相关是 音调检测、节拍检测和音频分析中的核心算法。</p>
     *
     * <h4>算法原理：</h4>
     * <ul>
     * <li><b>相关计算</b>：信号与其延迟版本的相关性</li>
     * <li><b>延迟范围</b>：从0到信号长度的一半</li>
     * <li><b>归一化</b>：按有效样本数进行平均</li>
     * </ul>
     *
     * <h4>数学公式：</h4>
     * <p>
     * R(τ) = (1/N) * Σ[x(n) * x(n+τ)]，其中τ为延迟量</p>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>基频(音调)检测</li>
     * <li>节拍和节奏分析</li>
     * <li>音频相似性度量</li>
     * <li>回声和混响检测</li>
     * </ul>
     *
     * @param signal 输入信号数据
     * @return 自相关函数值数组，长度为信号长度的一半
     */
    public static double[] calculateAutocorrelation(double[] signal) {
        int maxLag = signal.length / 2;
        double[] autocorr = new double[maxLag];

        for (int lag = 0; lag < maxLag; lag++) {
            double sum = 0;
            int count = 0;

            for (int i = 0; i < signal.length - lag; i++) {
                sum += signal[i] * signal[i + lag];
                count++;
            }

            autocorr[lag] = count > 0 ? sum / count : 0;
        }

        return autocorr;
    }

    /**
     * 计算RMS - 均方根值计算
     *
     * <p>
     * 计算音频信号的均方根(Root Mean Square)值，这是衡量信号整体 幅度和响度的重要指标。RMS值反映了信号的有效功率。</p>
     *
     * <h4>数学公式：</h4>
     * <p>
     * RMS = √[(1/N) * Σ(x²)]，其中N为样本数，x为样本值</p>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>响度感知</b>：与人耳感知的响度高度相关</li>
     * <li><b>功率度量</b>：反映信号的有效功率</li>
     * <li><b>动态特征</b>：描述音频的动态范围</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音量检测和归一化</li>
     * <li>音频分段和静音检测</li>
     * <li>动态范围分析</li>
     * <li>音频质量评估</li>
     * </ul>
     *
     * @param samples 音频样本向量
     * @return RMS值，范围通常在[0, 1]
     */
    public static double calculateRMS(IVector<Double> samples) {
        double sum = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            sum += sample * sample;
        }
        return Math.sqrt(sum / samples.length());
    }

    /**
     * 计算MFCC特征矩阵 - 处理整个音频序列 Calculate MFCC feature matrix - processes entire
     * audio sequence
     *
     * @param audioData 音频数据
     * @param parameters 参数配置
     * @param mfccCount MFCC系数数量
     * @return MFCC矩阵，每行代表一帧，每列代表一个MFCC系数
     */
    public static double[][] calculateMFCCFrames(AudioData audioData, Map<String, Object> parameters, int mfccCount) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = (Integer) parameters.getOrDefault("hopSize", DEFAULT_HOP_SIZE);

        return calculateMFCCFrames(audioData, mfccCount, windowSize, hopSize);
    }

    /**
     * 计算MFCC特征矩阵 - 处理整个音频序列 Calculate MFCC feature matrix - processes entire
     * audio sequence
     *
     * @param audioData 音频数据
     * @param mfccCount MFCC系数数量
     * @param windowSize
     * @param hopSize
     * @return MFCC矩阵，每行代表一帧，每列代表一个MFCC系数
     */
    public static double[][] calculateMFCCFrames(AudioData audioData, int mfccCount, int windowSize, int hopSize) throws AudioProcessingException {

        // 确保使用单声道音频进行MFCC计算
        AudioData monoAudioData = audioData;
        if (!audioData.isMono()) {
            // 将立体声转换为单声道
            monoAudioData = AudioProcessor.stereoToMono(audioData);
        }

        IVector<Double> samples = monoAudioData.getSamples();
        int numFrames = (samples.length() - windowSize) / hopSize + 1;

        if (numFrames <= 0) {
            throw new AudioProcessingException("Audio too short for frame analysis");
        }

        double[][] mfccMatrix = new double[numFrames][mfccCount];

        for (int frameIndex = 0; frameIndex < numFrames; frameIndex++) {
            int startSample = frameIndex * hopSize;
            int endSample = Math.min(startSample + windowSize, samples.length());

            // 提取当前帧
            double[] frameArray = new double[windowSize];
            for (int i = 0; i < windowSize; i++) {
                if (startSample + i < endSample) {
                    frameArray[i] = samples.get(startSample + i);
                } else {
                    frameArray[i] = 0.0; // 零填充
                }
            }
            IVector<Double> frame = Linalg.vector(frameArray);

            // 应用窗函数
            IVector<Double> windowed = applyWindow(frame, windowSize);

            // 转换为复数进行FFT
            Complex[] input = new Complex[windowed.length()];
            for (int i = 0; i < windowed.length(); i++) {
                input[i] = new Complex(windowed.get(i), 0.0);
            }
            // 零填充确保长度为2的幂
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);
            Complex[] spectrum = RereFFT.fft(paddedInput);

            // 关键修复：梅尔滤波器数量应该大于MFCC系数数量
            // 标准配置：使用26-40个梅尔滤波器，然后通过DCT提取前13个MFCC系数
            // 这样可以提供足够的频率分辨率
            int numMelFilters = Math.max(26, mfccCount * 2); // 至少26个，或MFCC数量的2倍
            double[] melEnergies = calculateMelFilters(spectrum, monoAudioData.getSampleRate(), numMelFilters);

            // 应用DCT变换 (离散余弦变换) 到对数梅尔能量
            // MFCC计算流程：功率谱 -> 梅尔滤波 -> 取对数 -> DCT变换 -> 取前N个系数
            double[] mfcc = new double[mfccCount];
            for (int i = 0; i < mfccCount; i++) {
                double sum = 0.0;
                for (int j = 0; j < melEnergies.length; j++) {
                    // melEnergies是线性能量值，这里取对数转换为对数能量
                    double logEnergy = Math.log(melEnergies[j]); // melEnergies已确保>=1e-10，安全
                    
                    // DCT-II公式: cos(π * i * (2*j + 1) / (2 * N))
                    // 其中 i 是MFCC系数索引，j 是梅尔滤波器索引，N 是滤波器数量
                    double dctCoeff = Math.cos(Math.PI * i * (2 * j + 1) / (2.0 * melEnergies.length));
                    sum += logEnergy * dctCoeff;
                }
                
                // DCT-II正交归一化系数
                // 第0个系数使用 sqrt(1/N)，其他系数使用 sqrt(2/N)
                if (i == 0) {
                    mfcc[i] = sum * Math.sqrt(1.0 / melEnergies.length);
                } else {
                    mfcc[i] = sum * Math.sqrt(2.0 / melEnergies.length);
                }
            }
            
            // 复制到输出矩阵
            System.arraycopy(mfcc, 0, mfccMatrix[frameIndex], 0, mfccCount);
        }
        return mfccMatrix;
    }

    
    public static IMatrix<Double> calculateMFCCMatrix(AudioData audioData, int windowSize) throws AudioProcessingException {
        double[][] mfccArray = calculateMFCCFrames(audioData, DEFAULT_MFCC_COUNT, windowSize, DEFAULT_HOP_SIZE);
        return Linalg.matrix(mfccArray);
    }
    
    /**
     * 
     * @param audioData
     * @param mfccCount
     * @param windowSize
     * @return
     * @throws AudioProcessingException 
     */
    public static IMatrix<Double> calculateMFCCMatrix(AudioData audioData, int mfccCount, int windowSize) throws AudioProcessingException {
        double[][] mfccArray = calculateMFCCFrames(audioData, mfccCount, windowSize, DEFAULT_HOP_SIZE);
        return Linalg.matrix(mfccArray);
    }

    /**
     * 计算MFCC特征矩阵并返回IMatrix格式 Calculate MFCC feature matrix and return as
     * IMatrix format
     */
    public static IMatrix<Double> calculateMFCCMatrix(AudioData audioData, int mfccCount, int windowSize, int hopSize) throws AudioProcessingException {
        double[][] mfccArray = calculateMFCCFrames(audioData, mfccCount, windowSize, hopSize);
        return Linalg.matrix(mfccArray);
    }

    /**
     *
     * @param audioData
     * @return
     * @throws AudioProcessingException
     */
    public static IMatrix<Double> calculateMFCCMatrix(AudioData audioData) throws AudioProcessingException {
        int mfccCount = 13;
        return Linalg.matrix(calculateMFCCFrames(audioData, mfccCount, DEFAULT_WINDOW_SIZE, DEFAULT_HOP_SIZE));
    }

    /**
     * 计算时间序列的Delta特征 - 正确的Delta计算 Calculate temporal delta features - correct
     * delta calculation
     *
     * @param mfccFrames MFCC帧矩阵
     * @return Delta特征矩阵
     */
    public static double[][] calculateTemporalDelta(double[][] mfccFrames) {
        if (mfccFrames.length < 3) {
            throw new IllegalArgumentException("Need at least 3 frames for delta calculation");
        }

        int numFrames = mfccFrames.length;
        int numCoeffs = mfccFrames[0].length;
        double[][] deltaMatrix = new double[numFrames][numCoeffs];

        for (int frame = 0; frame < numFrames; frame++) {
            for (int coeff = 0; coeff < numCoeffs; coeff++) {
                if (frame == 0) {
                    // 第一帧：使用前向差分
                    deltaMatrix[frame][coeff] = mfccFrames[1][coeff] - mfccFrames[0][coeff];
                } else if (frame == numFrames - 1) {
                    // 最后一帧：使用后向差分
                    deltaMatrix[frame][coeff] = mfccFrames[frame][coeff] - mfccFrames[frame - 1][coeff];
                } else {
                    // 中间帧：使用中心差分
                    deltaMatrix[frame][coeff] = (mfccFrames[frame + 1][coeff] - mfccFrames[frame - 1][coeff]) / 2.0;
                }
            }
        }

        return deltaMatrix;
    }

    /**
     * 计算时间序列的Delta特征 - IMatrix版本 Calculate temporal delta features - IMatrix
     * version
     */
    public static IMatrix<Double> calculateTemporalDelta(IMatrix<Double> mfccMatrix) {
        int numFrames = mfccMatrix.rows();
        int numCoeffs = mfccMatrix.cols();

        if (numFrames < 3) {
            throw new IllegalArgumentException("Need at least 3 frames for delta calculation");
        }

        Double[][] deltaData = new Double[numFrames][numCoeffs];

        for (int frame = 0; frame < numFrames; frame++) {
            for (int coeff = 0; coeff < numCoeffs; coeff++) {
                if (frame == 0) {
                    deltaData[frame][coeff] = mfccMatrix.get(1, coeff) - mfccMatrix.get(0, coeff);
                } else if (frame == numFrames - 1) {
                    deltaData[frame][coeff] = mfccMatrix.get(frame, coeff) - mfccMatrix.get(frame - 1, coeff);
                } else {
                    deltaData[frame][coeff] = (mfccMatrix.get(frame + 1, coeff) - mfccMatrix.get(frame - 1, coeff)) / 2.0;
                }
            }
        }

        return Linalg.matrix(deltaData);
    }

    /**
     * 计算矩阵每列的均值 - 用于MFCC特征统计 Calculate mean of each column - for MFCC feature
     * statistics
     */
    public static double[] calculateColumnMeans(double[][] matrix) {
        if (matrix.length == 0) {
            return new double[0];
        }

        int numCols = matrix[0].length;
        double[] means = new double[numCols];

        for (int col = 0; col < numCols; col++) {
            double sum = 0.0;
            for (int row = 0; row < matrix.length; row++) {
                sum += matrix[row][col];
            }
            means[col] = sum / matrix.length;
        }

        return means;
    }

    /**
     * 计算矩阵每列的方差 - 用于MFCC特征统计 Calculate variance of each column - for MFCC
     * feature statistics
     */
    public static double[] calculateColumnVariances(double[][] matrix) {
        if (matrix.length == 0) {
            return new double[0];
        }

        double[] means = calculateColumnMeans(matrix);
        int numCols = matrix[0].length;
        double[] variances = new double[numCols];

        for (int col = 0; col < numCols; col++) {
            double sumSquaredDiff = 0.0;
            for (int row = 0; row < matrix.length; row++) {
                double diff = matrix[row][col] - means[col];
                sumSquaredDiff += diff * diff;
            }
            variances[col] = sumSquaredDiff / matrix.length;
        }

        return variances;
    }

    /**
     * 计算零交叉率 - 信号过零点统计
     *
     * <p>
     * 计算音频信号的零交叉率(Zero Crossing Rate)，即信号穿越零点的 频率。这是区分有声音和无声音、音乐和语音的重要特征。</p>
     *
     * <h4>计算方法：</h4>
     * <ul>
     * <li><b>符号变化</b>：检测相邻样本的符号变化</li>
     * <li><b>归一化</b>：除以总样本数得到比率</li>
     * </ul>
     *
     * <h4>特征意义：</h4>
     * <ul>
     * <li><b>低ZCR</b>：音调性强，如音乐、元音</li>
     * <li><b>高ZCR</b>：噪声性强，如摩擦音、辅音</li>
     * <li><b>中等ZCR</b>：混合信号，如语音</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>语音/音乐分类</li>
     * <li>有声/无声检测</li>
     * <li>音频分段</li>
     * <li>噪声检测</li>
     * </ul>
     *
     * @param samples 音频样本向量
     * @return 零交叉率，范围在[0, 1]
     */
    public static double calculateZeroCrossingRate(IVector<Double> samples) {
        int crossings = 0;
        for (int i = 1; i < samples.length(); i++) {
            if ((samples.get(i) >= 0) != (samples.get(i - 1) >= 0)) {
                crossings++;
            }
        }
        return (double) crossings / samples.length();
    }

    /**
     * 计算能量 - 信号总能量计算
     *
     * <p>
     * 计算音频信号的总能量，即所有样本平方和。与RMS不同， 能量值不进行归一化，直接反映信号的绝对强度。</p>
     *
     * <h4>数学公式：</h4>
     * <p>
     * Energy = Σ(x²)，其中x为样本值</p>
     *
     * <h4>与RMS的区别：</h4>
     * <ul>
     * <li><b>能量</b>：绝对值，与信号长度相关</li>
     * <li><b>RMS</b>：相对值，已归一化</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音频活动检测</li>
     * <li>信号强度比较</li>
     * <li>能量包络提取</li>
     * <li>动态特征分析</li>
     * </ul>
     *
     * @param samples 音频样本向量
     * @return 信号总能量值
     */
    public static double calculateEnergy(IVector<Double> samples) {
        double sum = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            sum += sample * sample;
        }
        return sum;
    }

    /**
     * 计算幅度包络 / Calculate amplitude envelope
     */
    public static double[] calculateAmplitudeEnvelope(IVector<Double> samples) {
        int frameSize = 512;
        int numFrames = (samples.length() - frameSize) / (frameSize / 2) + 1;
        double[] envelope = new double[numFrames];

        for (int i = 0; i < numFrames; i++) {
            int start = i * frameSize / 2;
            double maxAmp = 0.0;

            for (int j = 0; j < frameSize && start + j < samples.length(); j++) {
                maxAmp = Math.max(maxAmp, Math.abs(samples.get(start + j)));
            }

            envelope[i] = maxAmp;
        }

        return envelope;
    }

    /**
     * 应用窗函数 / Apply window function
     *
     * <p>
     * 应用指定类型的窗函数到信号上。支持多种窗函数类型，包括汉宁窗、汉明窗、 布莱克曼窗等。这是进行FFT等频域分析的必要预处理步骤。</p>
     *
     * <h4>窗函数类型：</h4>
     * <ul>
     * <li><b>RECTANGULAR</b>：矩形窗（无窗）</li>
     * <li><b>HANNING</b>：汉宁窗</li>
     * <li><b>HAMMING</b>：汉明窗</li>
     * <li><b>BLACKMAN</b>：布莱克曼窗</li>
     * <li><b>KAISER</b>：凯泽窗</li>
     * <li><b>BARTLETT</b>：巴特利特窗</li>
     * <li><b>GAUSSIAN</b>：高斯窗</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>FFT变换前的数据预处理</li>
     * <li>频域分析算法的输入准备</li>
     * <li>分帧处理中的数据格式转换</li>
     * </ul>
     *
     * @param signal 输入信号向量
     * @param windowSize 窗口大小
     * @param windowType 窗函数类型
     * @return 应用窗函数后的信号向量
     */
    public static IVector<Double> applyWindow(IVector<Double> signal, int windowSize, SignalUtilities.WindowType windowType) {
        // 获取指定类型的窗函数
        IVector<Double> window = SignalUtilities.window(windowSize, windowType);

        int length = Math.min(signal.length(), windowSize);
        IVector<Double> windowed = Linalg.zeros(windowSize);

        for (int i = 0; i < length; i++) {
            windowed.set(i, signal.get(i) * window.get(i));
        }

        return windowed;
    }

    /**
     * 应用窗函数（默认使用汉明窗） / Apply window function (default Hamming window)
     *
     * <p>
     * 应用汉明窗函数到信号上。这是进行FFT等频域分析的必要预处理步骤。</p>
     *
     * @param signal 输入信号向量
     * @param windowSize 窗口大小
     * @return 应用窗函数后的信号向量
     */
    public static IVector<Double> applyWindow(IVector<Double> signal, int windowSize) {
        return applyWindow(signal, windowSize, SignalUtilities.WindowType.HAMMING);
    }

    /**
     * 计算频谱重心 - 频谱能量分布的重心频率
     *
     * <p>
     * 计算频谱的重心(质心)，即频谱能量分布的加权平均频率。 频谱重心是描述音色明亮度的重要特征，反映了频谱的整体分布特性。</p>
     *
     * <h4>数学公式：</h4>
     * <p>
     * Centroid = Σ(f * |X(f)|) / Σ|X(f)|，其中f为频率，|X(f)|为幅度</p>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>音色特征</b>：反映音色的明亮度和尖锐度</li>
     * <li><b>频谱分布</b>：描述能量在频域的集中程度</li>
     * <li><b>感知相关</b>：与人耳感知的音色明暗度相关</li>
     * </ul>
     *
     * <h4>特征解释：</h4>
     * <ul>
     * <li><b>高重心</b>：明亮、尖锐的音色(如小提琴、钢琴高音)</li>
     * <li><b>低重心</b>：温暖、厚重的音色(如大提琴、低音)</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音色分析和分类</li>
     * <li>乐器识别</li>
     * <li>音频均衡器设计</li>
     * <li>音乐情感分析</li>
     * </ul>
     *
     * @param spectrum 频谱复数数组
     * @param sampleRate 采样率
     * @param windowSize 窗口大小
     * @return 频谱重心频率(Hz)
     */
    public static double calculateSpectralCentroid(Complex[] spectrum, double sampleRate, int windowSize) {
        double weightedSum = 0.0;
        double magnitudeSum = 0.0;

        // 修复：使用spectrum.length而非windowSize计算频率
        // 原因：FFT后可能进行了零填充，spectrum.length才是实际的FFT长度
        // 频率公式：f = bin_index * sampleRate / fft_length
        for (int i = 0; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            // 修复前：double frequency = (i * sampleRate) / windowSize; // 错误！
            double frequency = (i * sampleRate) / spectrum.length; // 正确的频率计算
            weightedSum += frequency * magnitude;
            magnitudeSum += magnitude;
        }

        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0;
    }

    /**
     * 计算频谱滚降 - 频谱能量累积的截止频率
     *
     * <p>
     * 计算频谱滚降点，即累积能量达到总能量85%时对应的频率。 这个特征描述了频谱能量的分布范围，反映音色的丰富程度。</p>
     *
     * <h4>算法步骤：</h4>
     * <ul>
     * <li><b>能量计算</b>：计算频谱总能量</li>
     * <li><b>阈值设定</b>：设置85%能量阈值</li>
     * <li><b>累积搜索</b>：从低频开始累积能量</li>
     * <li><b>截止确定</b>：找到达到阈值的频率点</li>
     * </ul>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>频谱宽度</b>：反映有效频谱的带宽</li>
     * <li><b>音色丰富度</b>：高滚降点表示丰富的高频成分</li>
     * <li><b>噪声指标</b>：可用于区分音乐和噪声</li>
     * </ul>
     *
     * <h4>特征解释：</h4>
     * <ul>
     * <li><b>高滚降</b>：明亮、丰富的音色(如打击乐、摇滚)</li>
     * <li><b>低滚降</b>：柔和、单纯的音色(如人声、弦乐)</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音色分析和分类</li>
     * <li>音乐风格识别</li>
     * <li>音频压缩优化</li>
     * <li>噪声检测</li>
     * </ul>
     *
     * @param spectrum 频谱复数数组
     * @param sampleRate 采样率
     * @param windowSize 窗口大小
     * @return 频谱滚降频率(Hz)
     */
    public static double calculateSpectralRolloff(Complex[] spectrum, double sampleRate, int windowSize) {
        // 修复1：使用能量（幅度平方）而非幅度
        // 修复2：使用spectrum.length而非windowSize计算频率
        double totalEnergy = 0.0;
        for (int i = 0; i < spectrum.length / 2; i++) {
            // 修复前：totalEnergy += spectrum[i].magnitude(); // 错误：应使用能量而非幅度
            double magnitude = spectrum[i].magnitude();
            totalEnergy += magnitude * magnitude; // 正确：能量 = 幅度的平方
        }

        double threshold = 0.85 * totalEnergy;
        double cumulativeEnergy = 0.0;

        for (int i = 0; i < spectrum.length / 2; i++) {
            // 修复前：cumulativeEnergy += spectrum[i].magnitude(); // 错误
            double magnitude = spectrum[i].magnitude();
            cumulativeEnergy += magnitude * magnitude; // 正确：累积能量
            if (cumulativeEnergy >= threshold) {
                // 修复前：return (i * sampleRate) / windowSize; // 错误的频率计算
                return (i * sampleRate) / spectrum.length; // 正确的频率计算
            }
        }

        return sampleRate / 2.0;
    }

    /**
     * 计算频谱带宽 - 频谱能量分布的离散程度
     *
     * <p>
     * 计算频谱带宽，即频谱能量围绕重心的分散程度。这个特征描述了 频谱的集中度，反映音色的纯净度和复杂度。</p>
     *
     * <h4>数学公式：</h4>
     * <p>
     * Bandwidth = √[Σ((f - centroid)² * |X(f)|) / Σ|X(f)|]</p>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>频谱分散度</b>：衡量频谱能量的分布范围</li>
     * <li><b>音色复杂度</b>：反映音色的丰富程度</li>
     * <li><b>谐波结构</b>：描述谐波的分布特性</li>
     * </ul>
     *
     * <h4>特征解释：</h4>
     * <ul>
     * <li><b>窄带宽</b>：纯净、单一的音色(如正弦波、笛子)</li>
     * <li><b>宽带宽</b>：复杂、丰富的音色(如噪声、打击乐)</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音色纯净度分析</li>
     * <li>乐器分类识别</li>
     * <li>音频质量评估</li>
     * <li>噪声检测</li>
     * </ul>
     *
     * @param spectrum 频谱复数数组
     * @param sampleRate 采样率
     * @param windowSize 窗口大小
     * @param centroid 频谱重心频率
     * @return 频谱带宽(Hz)
     */
    public static double calculateSpectralBandwidth(Complex[] spectrum, double sampleRate, int windowSize, double centroid) {
        double weightedSum = 0.0;
        double magnitudeSum = 0.0;

        // 修复：使用spectrum.length而非windowSize计算频率
        for (int i = 0; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            // 修复前：double frequency = (i * sampleRate) / windowSize; // 错误！
            double frequency = (i * sampleRate) / spectrum.length; // 正确的频率计算
            double diff = frequency - centroid;
            weightedSum += diff * diff * magnitude;
            magnitudeSum += magnitude;
        }

        return magnitudeSum > 0 ? Math.sqrt(weightedSum / magnitudeSum) : 0.0;
    }

    /**
     * 计算频谱对比度 - 频谱峰谷差异的度量
     *
     * <p>
     * 计算频谱对比度，通过分析不同频段内的最大值和最小值差异来 衡量频谱的动态范围。这个特征反映了音色的丰富程度和层次感。</p>
     *
     * <h4>算法步骤：</h4>
     * <ul>
     * <li><b>频段划分</b>：将频谱分为6个子频段</li>
     * <li><b>峰谷检测</b>：在每个频段内找到最大值和最小值</li>
     * <li><b>对比计算</b>：计算log(max/min)作为对比度</li>
     * <li><b>平均归一化</b>：对所有频段的对比度求平均</li>
     * </ul>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>动态范围</b>：反映频谱的动态变化程度</li>
     * <li><b>音色层次</b>：描述音色的丰富程度和复杂度</li>
     * <li><b>谐波结构</b>：体现谐波的强弱对比</li>
     * </ul>
     *
     * <h4>特征解释：</h4>
     * <ul>
     * <li><b>高对比度</b>：丰富、层次分明的音色</li>
     * <li><b>低对比度</b>：平坦、单调的音色</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音色复杂度分析</li>
     * <li>音乐风格分类</li>
     * <li>音频质量评估</li>
     * <li>乐器识别</li>
     * </ul>
     *
     * @param spectrum 频谱复数数组
     * @return 频谱对比度值
     */
    public static double calculateSpectralContrast(Complex[] spectrum) {
        int numBands = 6;
        double contrast = 0.0;

        for (int band = 0; band < numBands; band++) {
            int startBin = (band * spectrum.length / 2) / numBands;
            int endBin = ((band + 1) * spectrum.length / 2) / numBands;

            double maxMagnitude = 0.0;
            double minMagnitude = Double.MAX_VALUE;

            for (int i = startBin; i < endBin; i++) {
                double magnitude = spectrum[i].magnitude();
                maxMagnitude = Math.max(maxMagnitude, magnitude);
                minMagnitude = Math.min(minMagnitude, magnitude);
            }

            if (minMagnitude > 0 && maxMagnitude > 0) {
                contrast += Math.log(maxMagnitude / minMagnitude);
            }
        }

        return contrast / numBands;
    }

    /**
     * 计算频谱平坦度 - 频谱分布的均匀程度
     *
     * <p>
     * 计算频谱平坦度，通过几何平均数与算术平均数的比值来衡量 频谱分布的均匀程度。这个特征用于区分音调性信号和噪声性信号。</p>
     *
     * <h4>数学公式：</h4>
     * <p>
     * Flatness = GeometricMean / ArithmeticMean</p>
     * <p>
     * 其中几何平均数 = (∏|X(f)|)^(1/N)，算术平均数 = (1/N)Σ|X(f)|</p>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>分布均匀度</b>：衡量频谱能量分布的平坦程度</li>
     * <li><b>音调性指标</b>：区分音调性和噪声性信号</li>
     * <li><b>谐波特征</b>：反映谐波结构的规律性</li>
     * </ul>
     *
     * <h4>特征解释：</h4>
     * <ul>
     * <li><b>高平坦度(接近1)</b>：白噪声、无规律频谱</li>
     * <li><b>低平坦度(接近0)</b>：音调性强、谐波明显</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音调性/噪声性分类</li>
     * <li>语音/音乐区分</li>
     * <li>音频质量检测</li>
     * <li>信号类型识别</li>
     * </ul>
     *
     * @param spectrum 频谱复数数组
     * @return 频谱平坦度值，范围[0, 1]
     */
    public static double calculateSpectralFlatness(Complex[] spectrum) {
        // 修复：使用对数方法计算几何平均数，避免数值溢出和错误的累乘方式
        // 原错误：geometricMean *= Math.pow(magnitude, 1.0 / (spectrum.length / 2 - 1));
        // 这种方式在每次迭代都进行开方，最终结果完全错误
        
        // 正确方法：使用对数求和，最后取指数
        // 几何平均数 = exp((1/N) * Σ(log(x_i)))
        double logSum = 0.0;
        double arithmeticMean = 0.0;
        int count = 0;

        for (int i = 1; i < spectrum.length / 2; i++) {
            double magnitude = spectrum[i].magnitude();
            if (magnitude > 0) {
                logSum += Math.log(magnitude); // 累加对数值
                arithmeticMean += magnitude;
                count++;
            }
        }

        if (count > 0) {
            // 几何平均数 = exp(平均对数值)
            double geometricMean = Math.exp(logSum / count);
            arithmeticMean /= count;
            
            // 频谱平坦度 = 几何平均数 / 算术平均数
            return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0.0;
        }

        return 0.0;
    }

    /**
     * 计算频谱流量 - 频谱变化的时间导数
     *
     * <p>
     * 计算频谱流量，通过分析相邻帧之间频谱幅度的变化来衡量 音频信号的动态特性。这个特征对于检测音乐中的起始点、 节拍和音色变化非常有效。</p>
     *
     * <h4>算法原理：</h4>
     * <ul>
     * <li><b>帧间差分</b>：计算相邻帧频谱的幅度差</li>
     * <li><b>正向变化</b>：只考虑增加的频谱能量</li>
     * <li><b>累积求和</b>：对所有频率分量的变化求和</li>
     * <li><b>时间平均</b>：对所有帧的流量值求平均</li>
     * </ul>
     *
     * <h4>数学公式：</h4>
     * <p>
     * Flux(t) = Σ max(0, |X(t,f)| - |X(t-1,f)|)</p>
     * <p>
     * 其中X(t,f)表示时间t、频率f处的频谱幅度</p>
     *
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>动态变化</b>：反映音频信号的时间变化程度</li>
     * <li><b>起始检测</b>：新音符或音色的开始</li>
     * <li><b>节奏感知</b>：音乐的律动和节拍</li>
     * </ul>
     *
     * <h4>应用场景：</h4>
     * <ul>
     * <li>音符起始点检测</li>
     * <li>节拍跟踪</li>
     * <li>音乐分段</li>
     * <li>动态特征分析</li>
     * </ul>
     *
     * @param audioData 音频数据
     * @param parameters 参数映射，包含窗口大小和跳跃大小
     * @return 平均频谱流量值
     * @throws AudioProcessingException 音频处理异常
     */
    public static double calculateSpectralFlux(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        int windowSize = (Integer) parameters.getOrDefault("windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = (Integer) parameters.getOrDefault("hopSize", DEFAULT_HOP_SIZE);
        IVector<Double> samples = audioData.getSamples();

        Complex[] prevSpectrum = null;
        double totalFlux = 0.0;
        int frameCount = 0;

        for (int i = 0; i < samples.length() - windowSize; i += hopSize) {
            IVector<Double> frame = Linalg.zeros(windowSize);
            for (int j = 0; j < windowSize && i + j < samples.length(); j++) {
                frame.set(j, samples.get(i + j));
            }

            IVector<Double> windowed = AudioUtil.applyWindow(frame, windowSize);

            // Convert IVector<Double> to Complex[] for FFT
            Complex[] input = new Complex[windowed.length()];
            for (int j = 0; j < windowed.length(); j++) {
                input[j] = new Complex(windowed.get(j), 0.0);
            }
            // 零填充确保长度为2的幂
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);
            Complex[] spectrum = RereFFT.fft(paddedInput);

            if (prevSpectrum != null) {
                double flux = 0.0;
                for (int k = 0; k < spectrum.length / 2; k++) {
                    double diff = spectrum[k].magnitude() - prevSpectrum[k].magnitude();
                    flux += Math.max(0, diff);
                }
                totalFlux += flux;
                frameCount++;
            }

            prevSpectrum = spectrum;
        }

        return frameCount > 0 ? totalFlux / frameCount : 0.0;
    }

    /**
     * 计算梅尔滤波器能量 / Calculate mel filter bank energies
     * 
     * <p>标准MFCC计算流程中的梅尔滤波步骤：
     * 1. 计算功率谱（幅度平方）
     * 2. 应用梅尔滤波器组
     * 3. 对每个滤波器的输出求和得到能量
     * 4. 返回线性能量值（由调用方决定是否取对数）</p>
     * 
     * @param spectrum 频谱复数数组（FFT结果）
     * @param sampleRate 采样率
     * @param filterCount 梅尔滤波器数量
     * @return 梅尔滤波器能量数组（线性能量值，未取对数）
     */
    public static double[] calculateMelFilters(Complex[] spectrum, double sampleRate, int filterCount) {
        int spectrumLength = spectrum.length / 2;

        // 创建梅尔滤波器组 / Create Mel filter bank
        double[][] melFilterBank = createMelFilterBank(filterCount, spectrumLength, sampleRate);

        // 应用滤波器组到功率谱 / Apply filter bank to power spectrum
        double[] melEnergies = new double[filterCount];
        for (int i = 0; i < filterCount; i++) {
            double energy = 0.0;
            for (int j = 0; j < spectrumLength; j++) {
                // 计算功率谱：幅度的平方
                double magnitude = spectrum[j].magnitude();
                double power = magnitude * magnitude;
                
                // 应用梅尔滤波器权重并累加
                energy += power * melFilterBank[i][j];
            }
            // 确保能量值不为0（避免后续log(0)问题）
            // 使用一个很小的正数作为下限
            melEnergies[i] = Math.max(energy, 1e-10);
        }

        return melEnergies;
    }

    /**
     * 创建梅尔滤波器组 / Create Mel filter bank
     * 
     * <p>梅尔滤波器组是一组三角形带通滤波器，在梅尔刻度上均匀分布。
     * 梅尔刻度是一种非线性频率刻度，更接近人耳对频率的感知特性。</p>
     * 
     * <h4>算法步骤：</h4>
     * <ol>
     * <li>确定梅尔频率范围：从0Hz到Nyquist频率（sampleRate/2）</li>
     * <li>在梅尔刻度上均匀分布filterCount+2个点</li>
     * <li>将梅尔点转换回Hz频率</li>
     * <li>将Hz频率转换为频谱bin索引</li>
     * <li>为每个滤波器创建三角形窗口（左侧上升，右侧下降）</li>
     * </ol>
     * 
     * <h4>梅尔刻度转换公式：</h4>
     * <ul>
     * <li><b>Hz到Mel</b>：mel = 2595 * log10(1 + hz/700)</li>
     * <li><b>Mel到Hz</b>：hz = 700 * (10^(mel/2595) - 1)</li>
     * </ul>
     *
     * @param filterCount 滤波器数量 / Number of filters
     * @param spectrumLength 频谱长度（FFT长度的一半）/ Spectrum length
     * @param sampleRate 采样率 / Sample rate
     * @return 梅尔滤波器组矩阵 [filterCount x spectrumLength]
     */
    private static double[][] createMelFilterBank(int filterCount, int spectrumLength, double sampleRate) {
        // 计算梅尔频率范围 / Calculate Mel frequency range
        double melLow = hzToMel(0);
        double melHigh = hzToMel(sampleRate / 2.0);

        // 在梅尔刻度上均匀分布滤波器 / Distribute filters uniformly on Mel scale
        double[] melPoints = new double[filterCount + 2];
        for (int i = 0; i < filterCount + 2; i++) {
            melPoints[i] = melLow + (melHigh - melLow) * i / (filterCount + 1);
        }

        // 转换回Hz刻度 / Convert back to Hz scale
        double[] hzPoints = new double[filterCount + 2];
        for (int i = 0; i < filterCount + 2; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }

        // 将Hz点转换为频谱bin索引 / Convert Hz points to spectrum bin indices
        double[] binPoints = new double[filterCount + 2];
        for (int i = 0; i < filterCount + 2; i++) {
            binPoints[i] = spectrumLength * hzPoints[i] / (sampleRate / 2.0);
        }

        // 创建滤波器组 / Create filter bank
        double[][] filterBank = new double[filterCount][spectrumLength];

        for (int i = 0; i < filterCount; i++) {
            int startBin = (int) Math.floor(binPoints[i]);
            int midBin = (int) Math.floor(binPoints[i + 1]);
            int endBin = (int) Math.floor(binPoints[i + 2]);

            // 确保索引在有效范围内 / Ensure indices are within valid range
            startBin = Math.max(0, startBin);
            midBin = Math.max(0, Math.min(spectrumLength - 1, midBin));
            endBin = Math.max(0, Math.min(spectrumLength - 1, endBin));

            // 确保滤波器有有效的范围 / Ensure filter has valid range
            if (startBin >= endBin) {
                // 如果滤波器范围无效，设置为零 / If filter range is invalid, set to zero
                continue;
            }

            // 创建上升沿 / Create rising edge
            double riseDenominator = binPoints[i + 1] - binPoints[i];
            if (riseDenominator > 1e-10) {  // 避免除零 / Avoid division by zero
                for (int j = startBin; j <= midBin && j < spectrumLength; j++) {
                    filterBank[i][j] = Math.max(0.0, Math.min(1.0, (j - binPoints[i]) / riseDenominator));
                }
            }

            // 创建下降沿 / Create falling edge
            double fallDenominator = binPoints[i + 2] - binPoints[i + 1];
            if (fallDenominator > 1e-10) {  // 避免除零 / Avoid division by zero
                for (int j = midBin + 1; j <= endBin && j < spectrumLength; j++) {
                    filterBank[i][j] = Math.max(0.0, Math.min(1.0, (binPoints[i + 2] - j) / fallDenominator));
                }
            }
        }

        return filterBank;
    }

    /**
     * 将Hz转换为Mel刻度 / Convert Hz to Mel scale
     * 
     * <p>Mel刻度是一种基于人耳音高感知的非线性频率刻度。
     * 在低频率范围，Mel刻度与Hz近乎线性；在高频率范围，增长越来越慢。</p>
     * 
     * <h4>公式由来：</h4>
     * <p>基于 Stevens 和 Volkmann (1940) 的实验研究，
     * 表达了人耳对不同频率的感知特性。</p>
     * 
     * <h4>数学公式：</h4>
     * <p>mel = 2595 * log10(1 + hz / 700)</p>
     *
     * @param hz Hz频率值
     * @return 对应的Mel频率值
     */
    private static double hzToMel(double hz) {
        // Stevens & Volkmann (1940) 的Mel刻度公式
        return 2595.0 * Math.log10(1 + hz / 700.0);
    }

    /**
     * 将Mel转换为Hz刻度 / Convert Mel to Hz scale
     * 
     * <p>这是hzToMel的逆运算，将Mel频率转换回Hz频率。
     * 在创建梅尔滤波器组时，需要在Mel刻度上均匀分布后，
     * 再转换回Hz频率以确定滤波器的位置。</p>
     * 
     * <h4>数学公式：</h4>
     * <p>hz = 700 * (10^(mel / 2595) - 1)</p>
     * 
     * <h4>推导：</h4>
     * <p>从 mel = 2595 * log10(1 + hz / 700) 可得：<br>
     * mel / 2595 = log10(1 + hz / 700)<br>
     * 10^(mel / 2595) = 1 + hz / 700<br>
     * hz = 700 * (10^(mel / 2595) - 1)</p>
     *
     * @param mel Mel频率值
     * @return 对应的Hz频率值
     */
    private static double melToHz(double mel) {
        // Mel到Hz的逆转换公式
        return 700.0 * (Math.pow(10, mel / 2595.0) - 1);
    }

    /**
     * 计算粗糙度 / Calculate roughness
     * 
     * <p>粗糙度（Roughness）是衡量音频信号频谱不规则性的指标。
     * 它反映了频谱在相邻频率之间的变化程度。</p>
     * 
     * <h4>算法原理：</h4>
     * <p>对于每个频率bin，计算其与相邻两个bin平均值的差异。
     * 差异越大，说明频谱越不平滑，粗糙度越高。</p>
     * 
     * <h4>公式：</h4>
     * <p>Roughness = (1/N) * Σ|X(i) - (X(i-1) + X(i+1))/2|</p>
     * 
     * <h4>物理意义：</h4>
     * <ul>
     * <li><b>高粗糙度</b>：频谱不平滑，可能是噪声或失真信号</li>
     * <li><b>低粗糙度</b>：频谱平滑，音色纯净、谐波结构良好</li>
     * </ul>
     * 
     * @param audioData 音频数据
     * @return 粗糙度值，值越大表示频谱越不平滑
     */
    public static double calculateRoughness(AudioData audioData) {
        try {
            // 基于频谱不规则性计算粗糙度
            Complex[] spectrum = AudioUtil.processFFT(audioData);

            double roughness = 0.0;
            // 遍历所有中间频率bin（排除第一个和最后一个）
            for (int i = 1; i < spectrum.length - 1; i++) {
                double current = spectrum[i].magnitude();
                double prev = spectrum[i - 1].magnitude();
                double next = spectrum[i + 1].magnitude();

                // 计算当前频率bin与相邻两个bin平均值的差异
                double localVariation = Math.abs(current - (prev + next) / 2.0);
                roughness += localVariation;
            }

            // 归一化：除以总的bin数量
            return roughness / spectrum.length;

        } catch (Exception e) {
            return 0.0;
        }
    }

    /**
     * 计算语音频率范围能量
     * 
     * <p>分析音频信号中语音特征频率范围（300Hz-3400Hz）的能量占比。
     * 这个范围是电话语音的标准带宽，也是人类语音的主要频率区域。</p>
     * 
     * @param audioData 音频数据
     * @return 语音频率范围能量占总能量的比例，范围[0, 1]
     */
    public static double calculateSpeechFrequencyEnergy(AudioData audioData) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData);

            double totalEnergy = 0.0;
            double speechEnergy = 0.0;

            // 语音主要频率范围：300Hz - 3400Hz（电话语音带宽）
            // FFT频率分辨率 = sampleRate / spectrum.length
            // 频率bin计算公式：bin = frequency / (sampleRate / spectrum.length)
            //                    = frequency * spectrum.length / sampleRate
            double freqResolution = audioData.getSampleRate() / (double) spectrum.length;
            int speechMinBin = (int) Math.round(300.0 / freqResolution);
            int speechMaxBin = (int) Math.round(3400.0 / freqResolution);
            
            // 确保bin索引在有效范围内
            speechMinBin = Math.max(0, Math.min(speechMinBin, spectrum.length / 2 - 1));
            speechMaxBin = Math.max(0, Math.min(speechMaxBin, spectrum.length / 2 - 1));

            // 计算总能量和语音频段能量
            for (int i = 0; i < spectrum.length / 2; i++) {
                double magnitude = spectrum[i].magnitude();
                double energy = magnitude * magnitude; // 能量 = 幅度平方
                totalEnergy += energy;

                if (i >= speechMinBin && i <= speechMaxBin) {
                    speechEnergy += energy;
                }
            }

            return totalEnergy > 0 ? speechEnergy / totalEnergy : 0.0;

        } catch (Exception e) {
            return 0.5;
        }
    }

    /**
     * 计算人声频率范围能量
     * 
     * <p>分析音频信号中人声基频范围（80Hz-1100Hz）的能量占比。
     * 男声基频约85-180Hz，女声基频约165-255Hz，共振峰频率可达1100Hz。</p>
     * 
     * @param audioData 音频数据
     * @return 人声频率范围能量占总能量的比例，范围[0, 1]
     */
    public static double calculateVocalFrequencyEnergy(AudioData audioData) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData);

            double totalEnergy = 0.0;
            double vocalEnergy = 0.0;

            // 人声主要频率范围：80Hz - 1100Hz
            // 包括基频范围（80-300Hz）和主要共振峰频率（300-1100Hz）
            // FFT频率分辨率 = sampleRate / spectrum.length
            // 频率bin计算公式：bin = frequency / freqResolution
            double freqResolution = audioData.getSampleRate() / (double) spectrum.length;
            int vocalMinBin = (int) Math.round(80.0 / freqResolution);
            int vocalMaxBin = (int) Math.round(1100.0 / freqResolution);
            
            // 确保bin索引在有效范围内
            vocalMinBin = Math.max(0, Math.min(vocalMinBin, spectrum.length / 2 - 1));
            vocalMaxBin = Math.max(0, Math.min(vocalMaxBin, spectrum.length / 2 - 1));

            // 计算总能量和人声频段能量
            for (int i = 0; i < spectrum.length / 2; i++) {
                double magnitude = spectrum[i].magnitude();
                double energy = magnitude * magnitude; // 能量 = 幅度平方
                totalEnergy += energy;

                if (i >= vocalMinBin && i <= vocalMaxBin) {
                    vocalEnergy += energy;
                }
            }

            return totalEnergy > 0 ? vocalEnergy / totalEnergy : 0.0;

        } catch (Exception e) {
            return 0.5;
        }
    }

    /**
     * 计算背景噪声水平 / Calculate background noise level
     * 
     * <p>通过分析音频信号中能量最低的段落来估计背景噪声水平。
     * 这种方法假设噪声是相对稳定的，而有用信号会有较大的能量波动。</p>
     * 
     * <h4>算法步骤：</h4>
     * <ol>
     * <li>将音频分为多个短段（默认0.1秒/段）</li>
     * <li>计算每段的平均能量</li>
     * <li>找出能量最小的段落</li>
     * <li>该段落的RMS值作为噪声水平估计</li>
     * </ol>
     * 
     * <h4>假设：</h4>
     * <ul>
     * <li>背景噪声相对稳定且能量较住</li>
     * <li>有用信号（语音、音乐）会引起能量增大</li>
     * <li>至少存在一段只含噪声的时间区间</li>
     * </ul>
     * 
     * @param audioData 音频数据
     * @return 背景噪声的RMS值，范围通常在[0, 1]
     */
    public static double calculateBackgroundNoise(AudioData audioData) {
        try {
            double[] samples = audioData.getSamples().toDoubleArray();

            // 计算信号的最小能量段落作为噪声估计
            int segmentSize = (int) (audioData.getSampleRate() / 10.0); // 0.1秒段落
            double minEnergy = Double.MAX_VALUE;

            // 滑动窗口遍历所有段落
            for (int i = 0; i < samples.length - segmentSize; i += segmentSize) {
                double energy = 0.0;
                // 计算当前段落的能量
                for (int j = i; j < i + segmentSize && j < samples.length; j++) {
                    energy += samples[j] * samples[j];
                }
                energy /= segmentSize; // 平均能量

                // 记录最小能量
                if (energy < minEnergy) {
                    minEnergy = energy;
                }
            }

            // 返回RMS值（平均能量的平方根）
            return Math.sqrt(minEnergy);

        } catch (Exception e) {
            return 0.0;
        }
    }

    /**
     * 计算FBank特征矩阵 - 处理整个音频序列
     * Calculate FBank feature matrix - processes entire audio sequence
     *
     * @param audioData 音频数据 / Audio data
     * @param fbankCount FBank系数数量 / Number of FBank coefficients
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return FBank矩阵，每行代表一帧，每列代表一个FBank系数 / FBank matrix, each row represents a frame, each column represents an FBank coefficient
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public static double[][] calculateFBankFrames(AudioData audioData, int fbankCount, int windowSize, int hopSize) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        int numFrames = (samples.length() - windowSize) / hopSize + 1;

        if (numFrames <= 0) {
            throw new AudioProcessingException("Audio too short for frame analysis");
        }

        double[][] fbankMatrix = new double[numFrames][fbankCount];

        for (int frameIndex = 0; frameIndex < numFrames; frameIndex++) {
            int startSample = frameIndex * hopSize;
            int endSample = Math.min(startSample + windowSize, samples.length());

            // 提取当前帧 / Extract current frame
            double[] frameArray = new double[windowSize];
            for (int i = 0; i < windowSize; i++) {
                if (startSample + i < endSample) {
                    frameArray[i] = samples.get(startSample + i);
                } else {
                    frameArray[i] = 0.0; // 零填充 / Zero padding
                }
            }
            IVector<Double> frame = Linalg.vector(frameArray);

            // 应用窗函数 / Apply window function
            IVector<Double> windowed = applyWindow(frame, windowSize);

            // 转换为复数进行FFT / Convert to complex for FFT
            Complex[] input = new Complex[windowed.length()];
            for (int i = 0; i < windowed.length(); i++) {
                input[i] = new Complex(windowed.get(i), 0.0);
            }
            // 零填充确保长度为2的幂 / Zero pad to ensure length is power of two
            Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);
            Complex[] spectrum = RereFFT.fft(paddedInput);

            // 计算梅尔滤波器组输出（FBank特征） / Calculate Mel filter bank output (FBank features)
            double[] fbankFilters = calculateMelFilters(spectrum, audioData.getSampleRate(), fbankCount);
            
            // 将结果复制到FBank矩阵 / Copy results to FBank matrix
            System.arraycopy(fbankFilters, 0, fbankMatrix[frameIndex], 0, fbankCount);
        }
        return fbankMatrix;
    }

    /**
     * 计算FBank特征矩阵并返回IMatrix格式
     * Calculate FBank feature matrix and return as IMatrix format
     *
     * @param audioData 音频数据 / Audio data
     * @param fbankCount FBank系数数量 / Number of FBank coefficients
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return FBank特征矩阵 / FBank feature matrix
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public static IMatrix<Double> calculateFBankMatrix(AudioData audioData, int fbankCount, int windowSize, int hopSize) throws AudioProcessingException {
        double[][] fbankArray = calculateFBankFrames(audioData, fbankCount, windowSize, hopSize);
        return Linalg.matrix(fbankArray);
    }

    /**
     * 计算FBank特征矩阵（使用默认参数）
     * Calculate FBank feature matrix (using default parameters)
     *
     * @param audioData 音频数据 / Audio data
     * @return FBank特征矩阵 / FBank feature matrix
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public static IMatrix<Double> calculateFBankMatrix(AudioData audioData) throws AudioProcessingException {
        int fbankCount = 26; // 默认FBank系数数量 / Default FBank coefficient count
        return Linalg.matrix(calculateFBankFrames(audioData, fbankCount, DEFAULT_WINDOW_SIZE, DEFAULT_HOP_SIZE));
    }

    /**
     * 计算FBank特征矩阵
     * Calculate FBank feature matrix
     *
     * @param audioData 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @return FBank特征矩阵 / FBank feature matrix
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public static IMatrix<Double> calculateFBankMatrix(AudioData audioData, int windowSize) throws AudioProcessingException {
        double[][] fbankArray = calculateFBankFrames(audioData, 26, windowSize, DEFAULT_HOP_SIZE);
        return Linalg.matrix(fbankArray);
    }

    /**
     * 计算FBank特征矩阵
     * Calculate FBank feature matrix
     *
     * @param audioData 音频数据 / Audio data
     * @param fbankCount FBank系数数量 / Number of FBank coefficients
     * @param windowSize 窗口大小 / Window size
     * @return FBank特征矩阵 / FBank feature matrix
     * @throws AudioProcessingException 音频处理异常 / Audio processing exception
     */
    public static IMatrix<Double> calculateFBankMatrix(AudioData audioData, int fbankCount, int windowSize) throws AudioProcessingException {
        double[][] fbankArray = calculateFBankFrames(audioData, fbankCount, windowSize, DEFAULT_HOP_SIZE);
        return Linalg.matrix(fbankArray);
    }

}
