# 音频处理示例 (Audio Processing Examples)

## 概述 / Overview

本文档提供了 `com.reremouse.lab.audio` 包的详细使用示例，涵盖从基础操作到高级应用的各个方面。推荐使用 `Audios` 工厂类和 `Audios` 类来进行音频操作。

This document provides detailed usage examples for the `com.reremouse.lab.audio` package, covering everything from basic operations to advanced applications. It is recommended to use the `Audios` factory class and `Audios` class for audio operations.

## 基础示例 / Basic Examples

### 音频文件读写和基本操作 / Audio File I/O and Basic Operations

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.audio.core.UnsupportedAudioFormatException;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.math.linalg.IVector;
import java.io.IOException;

public class AudioBasicExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/sample.wav");
            
            // 获取音频基本信息 / Get basic audio information
            System.out.println("采样率: " + audioData.getSampleRate() + " Hz");
            System.out.println("声道数: " + audioData.getChannels());
            System.out.println("位深度: " + audioData.getBitDepth() + " bits");
            System.out.println("时长: " + audioData.getDuration() + " 秒");
            System.out.println("格式: " + audioData.getFormat());
            
            // 基本音频处理 / Basic audio processing
            AudioData louderAudio = Audios.adjustVolume(audioData, 1.5);  // 增加50%音量
            AudioData normalizedAudio = Audios.normalize(audioData);       // 标准化
            AudioData monoAudio = Audios.toMono(audioData);               // 转为单声道
            
            // 保存处理后的音频 / Save processed audio
            Audios.writeAudio(louderAudio, "output/louder.wav");
            Audios.writeAudio(normalizedAudio, "output/normalized.wav");
            Audios.writeAudio(monoAudio, "output/mono.wav");
            
            System.out.println("音频处理完成！");
            
        } catch (IOException | UnsupportedAudioFormatException e) {
            System.err.println("音频处理错误: " + e.getMessage());
        }
    }
}
```

### 音频格式转换 / Audio Format Conversion

```java
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;

public class AudioFormatConversionExample {
    public static void main(String[] args) {
        try {
            // 读取WAV文件 / Read WAV file
            AudioData audioData = Audios.readAudio("input/sample.wav");
            
            // 转换为不同格式 / Convert to different formats
            Audios.writeAudio(audioData, "output/converted.mp3", AudioFormat.MP3);
            Audios.writeAudio(audioData, "output/converted.flac", AudioFormat.FLAC);
            Audios.writeAudio(audioData, "output/converted.ogg", AudioFormat.OGG);
            
            // 批量转换 / Batch conversion
            String[] inputFiles = {"file1.wav", "file2.wav", "file3.wav"};
            for (String inputFile : inputFiles) {
                AudioData audio = Audios.readAudio("input/" + inputFile);
                String outputFile = inputFile.replace(".wav", ".mp3");
                Audios.writeAudio(audio, "output/" + outputFile, AudioFormat.MP3);
            }
            
            System.out.println("格式转换完成！");
            
        } catch (Exception e) {
            System.err.println("格式转换错误: " + e.getMessage());
        }
    }
}
```

## 音频分析示例 / Audio Analysis Examples

### 频谱分析和音高检测 / Spectrum Analysis and Pitch Detection

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.audio.analysis.IAudioAnalyzer;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

public class AudioAnalysisExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/music.wav");
            
            // 频谱分析 / Spectrum analysis
            Tuple2<IVector<Double>, IVector<Double>> spectrum = Audios.spectrum(audioData);
            IVector<Double> frequencies = spectrum.getFirst();
            IVector<Double> magnitudes = spectrum.getSecond();
            
            System.out.println("频谱分析完成，频率点数: " + frequencies.length());
            
            // 自定义窗口大小的频谱分析 / Custom window size spectrum analysis
            Tuple2<IVector<Double>, IVector<Double>> customSpectrum = 
                Audios.spectrum(audioData, 2048, 0.5);
            
            // 音高检测 / Pitch detection
            double pitch = Audios.detectPitch(audioData);
            System.out.println("检测到的音高: " + pitch + " Hz");
            
            // STFT分析 / STFT analysis
            Tuple2<IVector<Double>, IVector<Double>> stft = Audios.stft(audioData);
            System.out.println("STFT分析完成");
            
            // 基本统计信息 / Basic statistics
            double rms = Audios.calculateRMS(audioData);
            double zcr = Audios.calculateZeroCrossingRate(audioData);
            double energy = Audios.calculateEnergy(audioData);
            
            System.out.println("RMS: " + rms);
            System.out.println("过零率: " + zcr);
            System.out.println("能量: " + energy);
            
            // 使用分析器对象 / Using analyzer objects
            IAudioAnalyzer spectrumAnalyzer = Audios.createSpectrumAnalyzer();
            IAudioAnalyzer pitchDetector = Audios.createPitchDetector();
            IAudioAnalyzer stftAnalyzer = Audios.createSTFTAnalyzer();
            
            // 执行分析 / Perform analysis
            // (具体分析方法请参考各分析器的文档)
            
        } catch (Exception e) {
            System.err.println("音频分析错误: " + e.getMessage());
        }
    }
}
```

### 音频特征提取 / Audio Feature Extraction

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.audio.feature.IAudioFeatureExtractor;
import com.reremouse.lab.audio.feature.AudioFeatureResult;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;

public class AudioFeatureExtractionExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/speech.wav");
            
            // 基本特征提取 / Basic feature extraction
            IVector<Double> features = Audios.extractFeatures(audioData);
            System.out.println("提取的特征维度: " + features.length());
            
            // MFCC特征提取 / MFCC feature extraction
            IMatrix<Double> mfcc = Audios.calculateMFCC(audioData);
            System.out.println("MFCC特征矩阵大小: " + mfcc.rows() + "x" + mfcc.cols());
            
            // 自定义参数的MFCC / Custom MFCC parameters
            IMatrix<Double> customMfcc = Audios.calculateMFCC(audioData, 13, 1024, 512);
            
            // FBank特征提取 / FBank feature extraction
            IMatrix<Double> fbank = Audios.calculateFBank(audioData);
            System.out.println("FBank特征矩阵大小: " + fbank.rows() + "x" + fbank.cols());
            
            // 频谱特征计算 / Spectral feature calculation
            double spectralCentroid = Audios.calculateSpectralCentroid(audioData, 1024);
            double spectralRolloff = Audios.calculateSpectralRolloff(audioData, 1024);
            double spectralBandwidth = Audios.calculateSpectralBandwidth(audioData, 1024);
            double spectralContrast = Audios.calculateSpectralContrast(audioData);
            double spectralFlatness = Audios.calculateSpectralFlatness(audioData);
            
            System.out.println("频谱质心: " + spectralCentroid);
            System.out.println("频谱滚降: " + spectralRolloff);
            System.out.println("频谱带宽: " + spectralBandwidth);
            System.out.println("频谱对比度: " + spectralContrast);
            System.out.println("频谱平坦度: " + spectralFlatness);
            
            // 使用特征提取器对象 / Using feature extractor object
            IAudioFeatureExtractor extractor = Audios.createStandardFeatureExtractor();
            AudioFeatureResult result = extractor.extractAudioFeatures(audioData);// 提取时域、频域、谱域综合音频特征 / Extract time-domain, frequency-domain, and spectral features
            Tuple2<List<String>, IVector<Double>> result.toNumericalFeatures(); // 转换为数值特征(特征名、特征的值)

            // 获取不同类型的特征 / Get different types of features
            TimeDomainFeatureResult timeDomainFeatures = result.getTimeDomainFeatures();
            FrequencyDomainFeatureResult frequencyDomainFeatures = result.getFrequencyDomainFeatures();
            SpectralFeatureResult spectralFeatures = result.getSpectralFeatures();


            
        } catch (Exception e) {
            System.err.println("特征提取错误: " + e.getMessage());
        }
    }
}
```

## 音频处理示例 / Audio Processing Examples

### 音频滤波和效果处理 / Audio Filtering and Effects Processing

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.audio.filter.IBaseAudioFilter;
import com.reremouse.lab.audio.effect.IAudioEffect;
import java.util.Map;
import java.util.HashMap;

public class AudioFilteringEffectsExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/music.wav");
            
            // 低通滤波 / Low-pass filtering
            AudioData filteredAudio = Audios.lowPassFilter(audioData, 1000.0); // 截止频率1000Hz
            Audios.writeAudio(filteredAudio, "output/filtered.wav");
            
            // 混响效果 / Reverb effect
            AudioData reverbAudio = Audios.reverb(audioData, 0.5, 0.3); // 衰减0.5，湿声混合0.3
            Audios.writeAudio(reverbAudio, "output/reverb.wav");
            
            // 使用滤波器对象 / Using filter objects
            IBaseAudioFilter lowPassFilter = Audios.createLowPassFilter();
            IBaseAudioFilter advancedLowPassFilter = Audios.createAdvancedLowPassFilter();
            
            // 使用效果器对象 / Using effect objects
            IAudioEffect reverbEffect = Audios.createReverbEffect();
            
            // 链式处理 / Chain processing
            AudioData processedAudio = audioData;
            processedAudio = Audios.lowPassFilter(processedAudio, 2000.0);  // 先滤波
            processedAudio = Audios.reverb(processedAudio, 0.4, 0.2);       // 再加混响
            processedAudio = Audios.normalize(processedAudio);              // 最后标准化
            
            Audios.writeAudio(processedAudio, "output/chain_processed.wav");
            
            System.out.println("音频滤波和效果处理完成！");
            
        } catch (Exception e) {
            System.err.println("音频处理错误: " + e.getMessage());
        }
    }
}
```

### 音频增强处理 / Audio Enhancement Processing

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.audio.enhancement.IAudioEnhancer;
import java.util.Map;
import java.util.HashMap;

public class AudioEnhancementExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/noisy_speech.wav");
            
            // 降噪处理 / Noise reduction
            AudioData denoisedAudio = Audios.reduceNoise(audioData, 0.1); // 阈值0.1
            Audios.writeAudio(denoisedAudio, "output/denoised.wav");
            
            // 均衡器处理 / Equalizer processing
            Map<String, Double> bandGains = new HashMap<>();
            bandGains.put("low", 1.2);    // 低频增益20%
            bandGains.put("mid", 1.0);    // 中频保持不变
            bandGains.put("high", 0.8);   // 高频衰减20%
            AudioData equalizedAudio = Audios.equalize(audioData, bandGains);
            Audios.writeAudio(equalizedAudio, "output/equalized.wav");
            
            // 压缩器处理 / Compressor processing
            AudioData compressedAudio = Audios.compress(audioData, 0.7, 4.0); // 阈值0.7，比率4:1
            Audios.writeAudio(compressedAudio, "output/compressed.wav");
            
            // 使用增强器对象 / Using enhancer objects
            IAudioEnhancer noiseReductionEnhancer = Audios.createNoiseReductionEnhancer();
            IAudioEnhancer equalizerEnhancer = Audios.createEqualizerEnhancer();
            IAudioEnhancer compressorEnhancer = Audios.createCompressorEnhancer();
            
            // 综合增强处理 / Comprehensive enhancement processing
            AudioData enhancedAudio = audioData;
            enhancedAudio = Audios.reduceNoise(enhancedAudio, 0.15);        // 降噪
            enhancedAudio = Audios.equalize(enhancedAudio, bandGains);      // 均衡
            enhancedAudio = Audios.compress(enhancedAudio, 0.6, 3.0);       // 压缩
            enhancedAudio = Audios.normalize(enhancedAudio);                // 标准化
            
            Audios.writeAudio(enhancedAudio, "output/enhanced.wav");
            
            System.out.println("音频增强处理完成！");
            
        } catch (Exception e) {
            System.err.println("音频增强错误: " + e.getMessage());
        }
    }
}
```

## 音频可视化示例 / Audio Visualization Examples

### 音频波形和频谱可视化 / Audio Waveform and Spectrum Visualization

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.AudioVisualizer;
import com.reremouse.lab.math.viz.IPlot;
import java.util.List;

public class AudioVisualizationExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/sample.wav");
            
            // 绘制波形图 / Plot waveform
            IPlot waveformPlot = AudioVisualizer.plotWaveform(audioData, "音频波形图");
            waveformPlot.show(); // 显示图表
            
            // 绘制频谱图 / Plot spectrum
            IPlot spectrumPlot = AudioVisualizer.plotSpectrum(audioData, "音频频谱图");
            spectrumPlot.show();
            
            // 绘制对数频谱图 / Plot log spectrum
            IPlot logSpectrumPlot = AudioVisualizer.plotLogSpectrum(audioData, "对数频谱图");
            logSpectrumPlot.show();
            
            // 绘制频谱图 / Plot spectrogram
            IPlot spectrogramPlot = AudioVisualizer.plotSpectrogram(audioData, "频谱图");
            spectrogramPlot.show();
            
            // 绘制MFCC特征图 / Plot MFCC features
            IPlot mfccPlot = AudioVisualizer.plotMFCC(audioData, "MFCC特征图");
            mfccPlot.show();
            
            // 绘制音频统计信息 / Plot audio statistics
            IPlot statisticsPlot = AudioVisualizer.plotAudioStatistics(audioData, "音频统计信息");
            statisticsPlot.show();
            
            // 绘制音频质量信息 / Plot audio quality
            IPlot qualityPlot = AudioVisualizer.plotAudioQuality(audioData, "音频质量分析");
            qualityPlot.show();
            
            // 创建音频仪表板 / Create audio dashboard
            List<IPlot> dashboard = AudioVisualizer.createAudioDashboard(audioData, "音频分析仪表板");
            for (IPlot plot : dashboard) {
                plot.show();
            }
            
            System.out.println("音频可视化完成！");
            
        } catch (Exception e) {
            System.err.println("音频可视化错误: " + e.getMessage());
        }
    }
}
```

### 音频对比可视化 / Audio Comparison Visualization

```java
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.AudioVisualizer;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.math.viz.IPlot;

public class AudioComparisonVisualizationExample {
    public static void main(String[] args) {
        try {
            // 读取原始音频和处理后音频 / Read original and processed audio
            AudioData originalAudio = Audios.readAudio("input/original.wav");
            AudioData processedAudio = Audios.normalize(originalAudio);
            
            // 对比可视化 / Comparison visualization
            IPlot comparisonPlot = AudioVisualizer.plotAudioComparison(
                originalAudio, processedAudio, "原始音频 vs 处理后音频");
            comparisonPlot.show();
            
            // 分别显示波形 / Show waveforms separately
            IPlot originalWaveform = AudioVisualizer.plotWaveform(originalAudio, "原始音频波形");
            IPlot processedWaveform = AudioVisualizer.plotWaveform(processedAudio, "处理后音频波形");
            
            originalWaveform.show();
            processedWaveform.show();
            
            // 分别显示频谱 / Show spectrums separately
            IPlot originalSpectrum = AudioVisualizer.plotSpectrum(originalAudio, "原始音频频谱");
            IPlot processedSpectrum = AudioVisualizer.plotSpectrum(processedAudio, "处理后音频频谱");
            
            originalSpectrum.show();
            processedSpectrum.show();
            
            System.out.println("音频对比可视化完成！");
            
        } catch (Exception e) {
            System.err.println("音频对比可视化错误: " + e.getMessage());
        }
    }
}
```

## 7. 音频向量嵌入训练 / Audio Vector Embedding Training

### 7.1 批量训练i-vector向量嵌入模型 / Batch Training of i-vector Embedding Model

```java
import com.reremouse.lab.audio.embedding.IVectorEmbedding;
import com.reremouse.lab.audio.embedding.IAudioEmbedding;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioUtil;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

public class IVectorTrainingExample {
    public static void main(String[] args) {
        try {
            System.out.println("=== 传统i-vector模型训练示例 / Traditional i-vector Model Training Example ===\n");
            
            // 1. 创建i-vector模型 / Create i-vector model
            int ivectorDim = 100;      // i-vector维度
            int numComponents = 32;    // UBM高斯分量数
            int featureDim = 13;       // MFCC特征维度
            
            IVectorEmbedding model = new IVectorEmbedding(ivectorDim, numComponents, featureDim);
            
            System.out.println("创建i-vector模型:");
            System.out.println("- i-vector维度: " + model.getEmbeddingDimension());
            System.out.println("- UBM高斯分量数: " + numComponents);
            System.out.println("- MFCC特征维度: " + featureDim);
            System.out.println();
            
            // 2. 准备训练数据 / Prepare training data
            System.out.println("准备训练数据...");
            List<IMatrix<Double>> trainingData = new ArrayList<>();
            
            // 音频文件列表 / Audio file list
            String[] audioFiles = {
                "data/speaker1_sample1.wav",
                "data/speaker1_sample2.wav", 
                "data/speaker2_sample1.wav",
                "data/speaker2_sample2.wav",
                "data/speaker3_sample1.wav"
            };
            
            // 加载所有音频文件的MFCC特征 / Load MFCC features from all audio files
            for (String audioFile : audioFiles) {
                try {
                    AudioData audioData = Audios.readAudio(audioFile);
                    IMatrix<Double> mfccFeatures = AudioUtil.calculateMFCCMatrix(audioData);
                    trainingData.add(mfccFeatures);
                    
                    System.out.println("- 加载音频文件: " + audioFile + 
                                     " (MFCC维度: " + mfccFeatures.getRowCount() + 
                                     "x" + mfccFeatures.getColumnCount() + ")");
                } catch (Exception e) {
                    System.out.println("- 跳过文件 " + audioFile + ": " + e.getMessage());
                }
            }
            
            System.out.println("训练数据统计:");
            System.out.println("- 音频文件数: " + trainingData.size());
            System.out.println();
            
            // 3. 训练模型 / Train model
            System.out.println("开始训练i-vector模型...");
            long startTime = System.currentTimeMillis();
            
            model.train(trainingData);
            
            long trainingTime = System.currentTimeMillis() - startTime;
            System.out.println("训练完成! 耗时: " + trainingTime + "ms");
            System.out.println();
            
            // 4. 提取i-vector特征 / Extract i-vector features
            System.out.println("提取i-vector特征...");
            
            // 测试音频文件 / Test audio files
            String[] testFiles = {
                "data/test_speaker1.wav",
                "data/test_speaker2.wav"
            };
            
            List<IVector<Double>> embeddings = new ArrayList<>();
            
            for (String testFile : testFiles) {
                try {
                    AudioData testAudio = Audios.readAudio(testFile);
                    IVector<Double> embedding = model.embed(testAudio);
                    embeddings.add(embedding);
                    
                    System.out.println("- 文件: " + testFile);
                    System.out.println("  i-vector维度: " + embedding.size());
                    System.out.println("  前5个特征值: " + 
                                     embedding.subVector(0, Math.min(5, embedding.size())));
                    
                } catch (Exception e) {
                    System.out.println("- 处理文件 " + testFile + " 失败: " + e.getMessage());
                }
            }
            
            // 5. 计算相似度 / Calculate similarity
            if (embeddings.size() >= 2) {
                System.out.println("\n计算嵌入向量相似度:");
                double similarity = embeddings.get(0).cosineSimilarity(embeddings.get(1));
                double distance = embeddings.get(0).euclideanDistance(embeddings.get(1));
                
                System.out.println("- 余弦相似度: " + String.format("%.4f", similarity));
                System.out.println("- 欧氏距离: " + String.format("%.4f", distance));
            }
            
        } catch (Exception e) {
            System.err.println("i-vector训练错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
```

### 7.2 增量训练大规模音频文件的i-Vector向量嵌入模型 / Incremental Training of i-Vector-based Embedding model for Large-scale Audio Files

```java
import com.reremouse.lab.audio.embedding.OnlineIVectorEmbedding;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioUtil;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

public class RealAudioEmbeddingExample {
    public static void main(String[] args) {
        try {
            System.out.println("=== 实际音频文件嵌入训练示例 / Real Audio File Embedding Training Example ===\n");
            
            // 1. 创建嵌入模型 / Create embedding model
            OnlineIVectorEmbedding model = new OnlineIVectorEmbedding(64, 24, 13, true, 0.001);
            
            // 2. 扫描音频文件目录 / Scan audio file directory
            String audioDir = "data/audio_samples";
            List<String> audioFiles = scanAudioFiles(audioDir);
            
            System.out.println("发现音频文件: " + audioFiles.size() + " 个");
            
            // 3. 循环处理和训练音频文件 / Process and train audio files iteratively
            Map<String, IVector<Double>> fileEmbeddings = new HashMap<>();
            
            System.out.println("开始循环处理和训练...");
            long startTime = System.currentTimeMillis();
            
            int processedFiles = 0;
            for (String audioFile : audioFiles) {
                try {
                    System.out.println("处理文件: " + new File(audioFile).getName());
                    
                    // 加载和预处理音频 / Load and preprocess audio
                    AudioData audioData = Audios.readAudio(audioFile);
                    audioData = Audios.normalize(audioData);
                    audioData = Audios.reduceNoise(audioData, 0.1);
                    
                    // 提取MFCC特征 / Extract MFCC features
                    IMatrix<Double> mfccFeatures = AudioUtil.calculateMFCCMatrix(audioData);
                    
                    System.out.println("- MFCC维度: " + mfccFeatures.rows() + 
                                     "x" + mfccFeatures.columns());
                    
                    // 增量训练模型 / Incremental training
                    model.incrementalTrain(mfccFeatures);
                    
                    // 生成嵌入向量 / Generate embedding
                    IVector<Double> embedding = model.embed(mfccFeatures);
                    fileEmbeddings.put(audioFile, embedding);
                    
                    processedFiles++;
                    System.out.println("- 训练进度: " + processedFiles + "/" + audioFiles.size());
                    System.out.println("- 嵌入向量维度: " + embedding.size());
                    
                    // 释放内存 / Free memory
                    audioData = null;
                    mfccFeatures = null;
                    embedding = null;
                    System.gc(); // 建议垃圾回收
                    
                } catch (Exception e) {
                    System.out.println("- 跳过文件 " + audioFile + ": " + e.getMessage());
                }
            }
            
            // 4. 完成训练 / Finish training
            model.finishTraining();
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.println("\n训练完成! 耗时: " + trainingTime + "ms");
            System.out.println("成功处理的文件数: " + processedFiles);
            
            // 6. 相似度矩阵计算 / Similarity matrix calculation
            System.out.println("\n计算相似度矩阵...");
            
            List<String> fileList = new ArrayList<>(fileEmbeddings.keySet());
            System.out.println("文件相似度矩阵:");
            
            for (int i = 0; i < fileList.size(); i++) {
                String file1 = fileList.get(i);
                
                for (int j = 0; j < fileList.size(); j++) {
                    String file2 = fileList.get(j);
                    
                    if (i == j) {
                        System.out.print("1.0000 ");
                    } else {
                        IVector<Double> emb1 = fileEmbeddings.get(file1);
                        IVector<Double> emb2 = fileEmbeddings.get(file2);
                        
                        double similarity = emb1.cosineSimilarity(emb2);
                        System.out.print(String.format("%.4f ", similarity));
                    }
                }
                System.out.println();
            }
            
        } catch (Exception e) {
            System.err.println("实际音频嵌入训练错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 扫描音频文件目录 / Scan audio file directory
     */
    private static List<String> scanAudioFiles(String directory) {
        List<String> audioFiles = new ArrayList<>();
        File dir = new File(directory);
        
        if (!dir.exists() || !dir.isDirectory()) {
            System.out.println("目录不存在，使用模拟数据: " + directory);
            // 返回模拟的文件路径用于演示
            audioFiles.add("data/speaker1_sample1.wav");
            audioFiles.add("data/speaker1_sample2.wav");
            audioFiles.add("data/speaker2_sample1.wav");
            return audioFiles;
        }
        
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile() && isAudioFile(file.getName())) {
                    audioFiles.add(file.getAbsolutePath());
                }
            }
        }
        
        return audioFiles;
    }
    
    /**
     * 检查是否为音频文件 / Check if it's an audio file
     */
    private static boolean isAudioFile(String filename) {
        String[] audioExtensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"};
        String lowerName = filename.toLowerCase();
        
        for (String ext : audioExtensions) {
            if (lowerName.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }
    

}
```

## 8. 高级应用示例 / Advanced Application Examples

### 8.1 音频分割 / Audio Segmentation and Concatenation

```java

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

public class AudioSegmentationExample {
    public static void main(String[] args) {
        try {
            // 读取长音频文件 / Read long audio file
            AudioData longAudio = Audios.readAudio("input/long_audio.wav");
            
            // 分割音频 / Segment audio
            double segmentDuration = 10.0; // 10秒片段
            int numSegments = (int) Math.ceil(longAudio.getDuration() / segmentDuration);
            
            for (int i = 0; i < numSegments; i++) {
                double startTime = i * segmentDuration;
                double endTime = Math.min((i + 1) * segmentDuration, longAudio.getDuration());
                
                AudioData segment = longAudio.extractSegment(startTime, endTime);
                Audios.writeAudio(segment, "output/segment_" + i + ".wav");
            }
            
            // 音频拼接 / Audio concatenation
            AudioData segment1 = Audios.readAudio("output/segment_0.wav");
            AudioData segment2 = Audios.readAudio("output/segment_1.wav");
            AudioData segment3 = Audios.readAudio("output/segment_2.wav");
            
            // 使用向量API拼接音频样本 / Concatenate audio samples using vector API
            IVector<Double> samples1 = segment1.getSamples();
            IVector<Double> samples2 = segment2.getSamples();
            IVector<Double> samples3 = segment3.getSamples();
            
            // 使用向量拼接操作 / Use vector concatenation operations
            IVector<Double> concatenatedSamples = samples1.concat(samples2).concat(samples3);
            
            // 创建拼接后的音频数据 / Create concatenated audio data
            AudioData concatenatedAudio = new AudioData(
                concatenatedSamples,
                segment1.getSampleRate(),
                segment1.getChannels(),
                segment1.getBitDepth(),
                segment1.getFormat()
            );
            
            Audios.writeAudio(concatenatedAudio, "output/concatenated.wav");
            
            System.out.println("音频分割和拼接完成！");
            
        } catch (Exception e) {
            System.err.println("音频分割拼接错误: " + e.getMessage());
        }
    }
}
```

### 8.2 音频混合和淡入淡出 / Audio Mixing and Fade Effects

```java

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;

public class AudioMixingFadeExample {
    public static void main(String[] args) {
        try {
            // 读取两个音频文件 / Read two audio files
            AudioData music = Audios.readAudio("input/background_music.wav");
            AudioData voice = Audios.readAudio("input/voice.wav");
            
            // 确保两个音频具有相同的采样率和声道数 / Ensure same sample rate and channels
            if (music.getSampleRate() != voice.getSampleRate()) {
                System.err.println("警告: 音频采样率不匹配");
            }
            
            // 调整音量 / Adjust volumes
            AudioData quietMusic = Audios.adjustVolume(music, 0.3);  // 背景音乐音量30%
            AudioData normalVoice = Audios.adjustVolume(voice, 1.0); // 人声音量100%
            
            // 音频混合 / Audio mixing
            AudioData mixedAudio = quietMusic.mixWith(normalVoice, 0.7); // 70%人声，30%音乐
            
            // 添加淡入淡出效果 / Add fade in/out effects
            AudioData fadeInAudio = mixedAudio.fadeIn(2.0);   // 2秒淡入
            AudioData finalAudio = fadeInAudio.fadeOut(3.0);  // 3秒淡出
            
            // 保存最终音频 / Save final audio
            Audios.writeAudio(finalAudio, "output/mixed_with_fade.wav");
            
            // 创建交叉淡化效果 / Create crossfade effect
            double crossfadeDuration = 1.0; // 1秒交叉淡化
            AudioData music1 = Audios.readAudio("input/song1.wav");
            AudioData music2 = Audios.readAudio("input/song2.wav");
            
            // 第一首歌淡出 / First song fade out
            AudioData song1FadeOut = music1.fadeOut(crossfadeDuration);
            
            // 第二首歌淡入 / Second song fade in
            AudioData song2FadeIn = music2.fadeIn(crossfadeDuration);
            
            // 计算重叠部分的混合 / Calculate overlap mixing
            double overlapStart = song1FadeOut.getDuration() - crossfadeDuration;
            AudioData song1Overlap = song1FadeOut.extractSegment(overlapStart, song1FadeOut.getDuration());
            AudioData song2Overlap = song2FadeIn.extractSegment(0, crossfadeDuration);
            
            AudioData crossfadeOverlap = song1Overlap.mixWith(song2Overlap, 0.5);
            
            System.out.println("音频混合和淡入淡出处理完成！");
            
        } catch (Exception e) {
            System.err.println("音频混合处理错误: " + e.getMessage());
        }
    }
}
```

### 8.3 批量音频处理 / Batch Audio Processing

```java

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioFormat;
import com.reremouse.lab.audio.Audios;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class BatchAudioProcessingExample {
    public static void main(String[] args) {
        try {
            // 获取输入目录中的所有音频文件 / Get all audio files in input directory
            File inputDir = new File("input/");
            File[] audioFiles = inputDir.listFiles((dir, name) -> 
                name.toLowerCase().endsWith(".wav") || 
                name.toLowerCase().endsWith(".mp3") ||
                name.toLowerCase().endsWith(".flac"));
            
            if (audioFiles == null || audioFiles.length == 0) {
                System.out.println("未找到音频文件");
                return;
            }
            
            // 创建输出目录 / Create output directory
            File outputDir = new File("output/batch_processed/");
            outputDir.mkdirs();
            
            // 并行处理音频文件 / Process audio files in parallel
            ExecutorService executor = Executors.newFixedThreadPool(4);
            List<Future<?>> futures = new ArrayList<>();
            
            for (File audioFile : audioFiles) {
                Future<?> future = executor.submit(() -> {
                    try {
                        processAudioFile(audioFile, outputDir);
                    } catch (Exception e) {
                        System.err.println("处理文件失败: " + audioFile.getName() + " - " + e.getMessage());
                    }
                });
                futures.add(future);
            }
            
            // 等待所有任务完成 / Wait for all tasks to complete
            for (Future<?> future : futures) {
                future.get();
            }
            
            executor.shutdown();
            System.out.println("批量音频处理完成！");
            
        } catch (Exception e) {
            System.err.println("批量处理错误: " + e.getMessage());
        }
    }
    
    private static void processAudioFile(File inputFile, File outputDir) throws Exception {
        System.out.println("处理文件: " + inputFile.getName());
        
        // 读取音频文件 / Read audio file
        AudioData audioData = Audios.readAudio(inputFile.getAbsolutePath());
        
        // 应用处理链 / Apply processing chain
        AudioData processedAudio = audioData;
        
        // 1. 标准化 / Normalize
        processedAudio = Audios.normalize(processedAudio);
        
        // 2. 降噪 / Noise reduction
        processedAudio = Audios.reduceNoise(processedAudio, 0.1);
        
        // 3. 低通滤波 / Low-pass filter
        processedAudio = Audios.lowPassFilter(processedAudio, 8000.0);
        
        // 4. 转换为单声道 / Convert to mono
        processedAudio = Audios.toMono(processedAudio);
        
        // 5. 添加淡入淡出 / Add fade in/out
        processedAudio = processedAudio.fadeIn(0.1);
        processedAudio = processedAudio.fadeOut(0.1);
        
        // 保存处理后的文件 / Save processed file
        String outputFileName = "processed_" + inputFile.getName();
        String outputPath = new File(outputDir, outputFileName).getAbsolutePath();
        
        // 转换为WAV格式 / Convert to WAV format
        Audios.writeAudio(processedAudio, outputPath.replace(getFileExtension(outputPath), ".wav"), 
                          AudioFormat.WAV);
        
        System.out.println("完成处理: " + inputFile.getName());
    }
    
    private static String getFileExtension(String fileName) {
        int lastDotIndex = fileName.lastIndexOf('.');
        return lastDotIndex > 0 ? fileName.substring(lastDotIndex) : "";
    }
}
```

### 8.4 音频质量分析和报告 / Audio Quality Analysis and Reporting

```java

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioStatistics;
import com.reremouse.lab.audio.core.AudioQuality;
import com.reremouse.lab.audio.Audios;
import com.reremouse.lab.audio.AudioVisualizer;
import com.reremouse.lab.math.viz.IPlot;
import java.io.FileWriter;
import java.io.PrintWriter;

public class AudioQualityAnalysisExample {
    public static void main(String[] args) {
        try {
            // 读取音频文件 / Read audio file
            AudioData audioData = Audios.readAudio("input/test_audio.wav");
            
            // 获取音频统计信息 / Get audio statistics
            AudioStatistics stats = audioData.getStatistics();
            
            // 计算各种音频质量指标 / Calculate various audio quality metrics
            double rms = Audios.calculateRMS(audioData);
            double zcr = Audios.calculateZeroCrossingRate(audioData);
            double energy = Audios.calculateEnergy(audioData);
            double spectralCentroid = Audios.calculateSpectralCentroid(audioData, 1024);
            double spectralRolloff = Audios.calculateSpectralRolloff(audioData, 1024);
            double spectralBandwidth = Audios.calculateSpectralBandwidth(audioData, 1024);
            double spectralContrast = Audios.calculateSpectralContrast(audioData);
            double spectralFlatness = Audios.calculateSpectralFlatness(audioData);
            
            // 生成质量报告 / Generate quality report
            generateQualityReport(audioData, stats, rms, zcr, energy, 
                                spectralCentroid, spectralRolloff, spectralBandwidth,
                                spectralContrast, spectralFlatness);
            
            // 生成可视化报告 / Generate visualization report
            generateVisualizationReport(audioData);
            
            System.out.println("音频质量分析完成！");
            
        } catch (Exception e) {
            System.err.println("音频质量分析错误: " + e.getMessage());
        }
    }
    
    private static void generateQualityReport(AudioData audioData, AudioStatistics stats,
                                            double rms, double zcr, double energy,
                                            double spectralCentroid, double spectralRolloff,
                                            double spectralBandwidth, double spectralContrast,
                                            double spectralFlatness) throws Exception {
        
        try (PrintWriter writer = new PrintWriter(new FileWriter("output/quality_report.txt"))) {
            writer.println("音频质量分析报告");
            writer.println("================");
            writer.println();
            
            // 基本信息 / Basic information
            writer.println("基本信息:");
            writer.println("  文件格式: " + audioData.getFormat());
            writer.println("  采样率: " + audioData.getSampleRate() + " Hz");
            writer.println("  声道数: " + audioData.getChannels());
            writer.println("  位深度: " + audioData.getBitDepth() + " bits");
            writer.println("  时长: " + String.format("%.2f", audioData.getDuration()) + " 秒");
            writer.println("  样本数: " + audioData.getLength());
            writer.println();
            
            // 幅度统计 / Amplitude statistics
            writer.println("幅度统计:");
            writer.println("  最大幅度: " + String.format("%.6f", audioData.getMaxAmplitude()));
            writer.println("  最小幅度: " + String.format("%.6f", audioData.getMinAmplitude()));
            writer.println("  RMS幅度: " + String.format("%.6f", audioData.getRMSAmplitude()));
            writer.println("  平均幅度: " + String.format("%.6f", audioData.getAverageAmplitude()));
            writer.println();
            
            // 音频特征 / Audio features
            writer.println("音频特征:");
            writer.println("  RMS: " + String.format("%.6f", rms));
            writer.println("  过零率: " + String.format("%.6f", zcr));
            writer.println("  能量: " + String.format("%.6f", energy));
            writer.println();
            
            // 频谱特征 / Spectral features
            writer.println("频谱特征:");
            writer.println("  频谱质心: " + String.format("%.2f", spectralCentroid) + " Hz");
            writer.println("  频谱滚降: " + String.format("%.2f", spectralRolloff) + " Hz");
            writer.println("  频谱带宽: " + String.format("%.2f", spectralBandwidth) + " Hz");
            writer.println("  频谱对比度: " + String.format("%.6f", spectralContrast));
            writer.println("  频谱平坦度: " + String.format("%.6f", spectralFlatness));
            writer.println();
            
            // 质量评估 / Quality assessment
            writer.println("质量评估:");
            if (audioData.getMaxAmplitude() > 0.95) {
                writer.println("  警告: 可能存在削波失真");
            }
            if (rms < 0.01) {
                writer.println("  警告: 音频信号过小");
            }
            if (spectralFlatness > 0.8) {
                writer.println("  警告: 可能包含过多噪声");
            }
            
            writer.println("报告生成完成。");
        }
    }
    
    private static void generateVisualizationReport(AudioData audioData) throws Exception {
        // 生成各种可视化图表 / Generate various visualization charts
        IPlot waveformPlot = AudioVisualizer.plotWaveform(audioData, "音频波形");
        IPlot spectrumPlot = AudioVisualizer.plotSpectrum(audioData, "频谱分析");
        IPlot spectrogramPlot = AudioVisualizer.plotSpectrogram(audioData, "频谱图");
        IPlot statisticsPlot = AudioVisualizer.plotAudioStatistics(audioData, "统计信息");
        IPlot qualityPlot = AudioVisualizer.plotAudioQuality(audioData, "质量分析");
        
        // 保存图表 / Save charts
        // (具体保存方法取决于可视化库的实现)
        
        System.out.println("可视化报告生成完成");
    }
}
```

## 9. 性能优化示例 / Performance Optimization Examples

### 9.1 内存优化处理 / Memory Optimized Processing

```java

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.Audios;

public class MemoryOptimizedProcessingExample {
    public static void main(String[] args) {
        try {
            // 处理大音频文件的内存优化方法 / Memory optimized method for large audio files
            String inputFile = "input/large_audio.wav";
            String outputFile = "output/processed_large_audio.wav";
            
            // 分块处理大文件 / Process large files in chunks
            processLargeAudioInChunks(inputFile, outputFile, 10.0); // 10秒块
            
            System.out.println("大文件处理完成！");
            
        } catch (Exception e) {
            System.err.println("大文件处理错误: " + e.getMessage());
        }
    }
    
    private static void processLargeAudioInChunks(String inputFile, String outputFile, 
                                                 double chunkDuration) throws Exception {
        // 读取音频文件头信息 / Read audio file header
        AudioData fullAudio = Audios.readAudio(inputFile);
        double totalDuration = fullAudio.getDuration();
        
        // 计算块数 / Calculate number of chunks
        int numChunks = (int) Math.ceil(totalDuration / chunkDuration);
        
        // 创建临时文件列表 / Create temporary file list
        List<String> tempFiles = new ArrayList<>();
        
        // 分块处理 / Process in chunks
        for (int i = 0; i < numChunks; i++) {
            double startTime = i * chunkDuration;
            double endTime = Math.min((i + 1) * chunkDuration, totalDuration);
            
            // 提取块 / Extract chunk
            AudioData chunk = fullAudio.extractSegment(startTime, endTime);
            
            // 处理块 / Process chunk
            AudioData processedChunk = processAudioChunk(chunk);
            
            // 保存临时文件 / Save temporary file
            String tempFile = "temp/chunk_" + i + ".wav";
            tempFiles.add(tempFile);
            Audios.writeAudio(processedChunk, tempFile);
            
            // 释放内存 / Free memory
            chunk = null;
            processedChunk = null;
            System.gc(); // 建议垃圾回收
            
            System.out.println("处理完成块 " + (i + 1) + "/" + numChunks);
        }
        
        // 合并临时文件 / Merge temporary files
        mergeAudioFiles(tempFiles, outputFile);
        
        // 清理临时文件 / Clean up temporary files
        tempFiles.forEach(tempFile -> new java.io.File(tempFile).delete());
    }
    
    private static AudioData processAudioChunk(AudioData chunk) {
        // 应用处理链 / Apply processing chain
        AudioData processed = chunk;
        processed = Audios.normalize(processed);
        processed = Audios.lowPassFilter(processed, 8000.0);
        return processed;
    }
    
    private static void mergeAudioFiles(List<String> inputFiles, String outputFile) throws Exception {
        // 简化的文件合并实现 / Simplified file merging implementation
        // 实际实现中应该使用更高效的流式合并方法
        
        if (!inputFiles.isEmpty()) {
            AudioData firstChunk = Audios.readAudio(inputFiles.get(0));
            // ... 合并逻辑
            Audios.writeAudio(firstChunk, outputFile);
        }
    }
}
```

这些示例展示了音频包的主要功能和使用方法，涵盖了从基础操作到高级应用的各个方面。用户可以根据具体需求选择合适的示例作为起点，并根据实际情况进行调整和扩展。