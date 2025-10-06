# 音频操作 (Audio Operations)

## 概述 / Overview

`com.yishape.lab.audio` 包提供了完整的音频处理功能，包括音频文件读写、音频分析、特征提取、音频处理、音频效果、音频可视化等。该包采用模块化设计，支持多种音频格式，提供丰富的音频处理和分析功能。

The `com.yishape.lab.audio` package provides comprehensive audio processing capabilities, including audio file I/O, audio analysis, feature extraction, audio processing, audio effects, and audio visualization. The package uses modular design, supports multiple audio formats, and offers rich audio processing and analysis functionalities.

## 核心接口 / Core Interface

### Audios 工厂类 / Audios Factory Class

`Audios` 类是音频处理的主要入口点，提供了统一的音频处理接口，封装了音频处理、分析、滤波、效果等核心功能。推荐使用此类来创建和操作音频相关组件。

`Audios` class is the main entry point for audio processing, providing a unified audio processing interface that encapsulates core functions such as audio processing, analysis, filtering, and effects. It is recommended to use this class to create and operate audio-related components.

**架构设计 / Architecture Design:**
- **工厂模式** / **Factory pattern**: `Audios` → `AudioComponentFactory` → 具体实现类
- **模块化设计** / **Modular design**: 分析、处理、效果、滤波等功能模块独立
- **API统一** / **API consistency**: 提供一致的命名和使用模式

### AudioData 数据类 / AudioData Data Class

`AudioData` 类封装了音频数据的基本信息，包括采样率、声道数、位深度、时长等，使用 `IVector<Double>` 接口存储音频样本数据，确保与现有代码库的兼容性。

`AudioData` class encapsulates basic audio information including sample rate, channels, bit depth, duration, etc. It uses `IVector<Double>` interface to store audio sample data, ensuring compatibility with existing codebase.

### AudioIO 输入输出类 / AudioIO Input/Output Class

`AudioIO` 类提供音频文件的读取和写入功能，支持多种音频格式，包括 WAV、MP3、FLAC、OGG 等。

`AudioIO` class provides audio file reading and writing functionality, supporting multiple audio formats including WAV, MP3, FLAC, OGG, etc.

## 主要功能 / Main Features

### 1. 音频文件操作 / Audio File Operations

#### 音频文件读取 / Audio File Reading

```java
// 推荐使用 AudioIO 类进行音频文件操作 / Recommended to use AudioIO class for audio file operations

import core.audio.com.yishape.lab.AudioIO;
import core.audio.com.yishape.lab.AudioData;
import core.audio.com.yishape.lab.AudioFormat;

// 自动识别格式读取 / Auto-detect format and read
AudioData audio = AudioIO.readAudio("path/to/audio.wav");

// 指定格式读取 / Read with specified format
AudioData audio = AudioIO.readAudio("path/to/audio.mp3", AudioFormat.MP3);
```

#### 音频文件写入 / Audio File Writing

```java
// 自动识别格式写入 / Auto-detect format and write
AudioIO.writeAudio(audioData, "output/path/audio.wav");

// 指定格式写入 / Write with specified format
AudioIO.writeAudio(audioData, "output/path/audio.mp3", AudioFormat.MP3);
```

### 2. 音频处理 / Audio Processing

#### 音频处理器创建 / Audio Processor Creation

```java
import audio.com.yishape.lab.Audios;
import processing.audio.com.yishape.lab.IAdvancedAudioProcessor;

// 创建音量处理器 / Create volume processor
IAdvancedAudioProcessor volumeProcessor = Audios.createVolumeProcessor();

// 创建标准化处理器 / Create normalize processor
IAdvancedAudioProcessor normalizeProcessor = Audios.createNormalizeProcessor();

// 创建声道处理器 / Create channel processor
IAdvancedAudioProcessor channelProcessor = Audios.createChannelProcessor();
```

#### 基本音频处理 / Basic Audio Processing

```java
// 调整音量 / Adjust volume
AudioData louderAudio = Audios.adjustVolume(audioData, 1.5); // 增加50%音量

// 标准化音频 / Normalize audio
AudioData normalizedAudio = Audios.normalize(audioData);

// 转换为单声道 / Convert to mono
AudioData monoAudio = Audios.toMono(audioData);

// 转换为立体声 / Convert to stereo
AudioData stereoAudio = Audios.toStereo(audioData);

// 转换声道数 / Convert channel count
AudioData convertedAudio = Audios.convertChannels(audioData, 2);
```

### 3. 音频分析 / Audio Analysis

#### 音频分析器创建 / Audio Analyzer Creation

```java
import analysis.audio.com.yishape.lab.IAudioAnalyzer;

// 创建频谱分析器 / Create spectrum analyzer
IAudioAnalyzer spectrumAnalyzer = Audios.createSpectrumAnalyzer();

// 创建音高检测器 / Create pitch detector
IAudioAnalyzer pitchDetector = Audios.createPitchDetector();

// 创建STFT分析器 / Create STFT analyzer
IAudioAnalyzer stftAnalyzer = Audios.createSTFTAnalyzer();
```

#### 基本音频分析 / Basic Audio Analysis

```java
import linalg.math.com.yishape.lab.IVector;
import util.com.yishape.lab.Tuple2;

// 频谱分析 / Spectrum analysis
Tuple2<IVector<Double>, IVector<Double>> spectrum = Audios.spectrum(audioData);
IVector<Double> frequencies = spectrum.getFirst();
IVector<Double> magnitudes = spectrum.getSecond();

// 音高检测 / Pitch detection
double pitch = Audios.detectPitch(audioData);

// STFT分析 / STFT analysis
Tuple2<IVector<Double>, IVector<Double>> stft = Audios.stft(audioData);

// 计算RMS / Calculate RMS
double rms = Audios.calculateRMS(audioData);

// 计算过零率 / Calculate zero crossing rate
double zcr = Audios.calculateZeroCrossingRate(audioData);

// 计算能量 / Calculate energy
double energy = Audios.calculateEnergy(audioData);
```

### 4. 音频特征提取 / Audio Feature Extraction

#### 特征提取器创建 / Feature Extractor Creation

```java
import feature.audio.com.yishape.lab.IAudioFeatureExtractor;

// 创建标准特征提取器 / Create standard feature extractor
IAudioFeatureExtractor featureExtractor = Audios.createStandardFeatureExtractor();
```

#### 基本特征提取 / Basic Feature Extraction

```java
import linalg.math.com.yishape.lab.IMatrix;

// 提取基本特征向量 / Extract basic feature vector
IVector<Double> features = Audios.extractFeatures(audioData);

// 计算MFCC特征 / Calculate MFCC features
IMatrix<Double> mfcc = Audios.calculateMFCC(audioData);
IMatrix<Double> mfccCustom = Audios.calculateMFCC(audioData, 13, 1024, 512);

// 计算FBank特征 / Calculate FBank features
IMatrix<Double> fbank = Audios.calculateFBank(audioData);
IMatrix<Double> fbankCustom = Audios.calculateFBank(audioData, 26, 1024, 512);

// 计算频谱特征 / Calculate spectral features
double spectralCentroid = Audios.calculateSpectralCentroid(audioData, 1024);
double spectralRolloff = Audios.calculateSpectralRolloff(audioData, 1024);
double spectralBandwidth = Audios.calculateSpectralBandwidth(audioData, 1024);
double spectralContrast = Audios.calculateSpectralContrast(audioData);
double spectralFlatness = Audios.calculateSpectralFlatness(audioData);
```

### 5. 音频滤波 / Audio Filtering

#### 音频滤波器创建 / Audio Filter Creation

```java
import filter.audio.com.yishape.lab.IBaseAudioFilter;

// 创建低通滤波器 / Create low-pass filter
IBaseAudioFilter lowPassFilter = Audios.createLowPassFilter();

// 创建高级低通滤波器 / Create advanced low-pass filter
IBaseAudioFilter advancedLowPassFilter = Audios.createAdvancedLowPassFilter();
```

#### 基本滤波操作 / Basic Filtering Operations

```java
// 低通滤波 / Low-pass filtering
AudioData filteredAudio = Audios.lowPassFilter(audioData, 1000.0); // 截止频率1000Hz
```

### 6. 音频效果 / Audio Effects

#### 音频效果器创建 / Audio Effect Creation

```java
import effect.audio.com.yishape.lab.IAudioEffect;

// 创建混响效果器 / Create reverb effect
IAudioEffect reverbEffect = Audios.createReverbEffect();
```

#### 基本效果处理 / Basic Effect Processing

```java
// 添加混响效果 / Add reverb effect
AudioData reverbAudio = Audios.reverb(audioData, 0.5, 0.3); // 衰减0.5，湿声混合0.3
```

### 7. 音频增强 / Audio Enhancement

#### 音频增强器创建 / Audio Enhancer Creation

```java
import enhancement.audio.com.yishape.lab.IAudioEnhancer;

// 创建降噪增强器 / Create noise reduction enhancer
IAudioEnhancer noiseReductionEnhancer = Audios.createNoiseReductionEnhancer();

// 创建均衡器增强器 / Create equalizer enhancer
IAudioEnhancer equalizerEnhancer = Audios.createEqualizerEnhancer();

// 创建压缩器增强器 / Create compressor enhancer
IAudioEnhancer compressorEnhancer = Audios.createCompressorEnhancer();
```

#### 基本增强处理 / Basic Enhancement Processing

```java
import java.util.Map;
import java.util.HashMap;

// 降噪处理 / Noise reduction
AudioData denoisedAudio = Audios.reduceNoise(audioData, 0.1); // 阈值0.1

// 均衡器处理 / Equalizer processing
Map<String, Double> bandGains = new HashMap<>();
bandGains.put("low", 1.2);    // 低频增益
bandGains.put("mid", 1.0);    // 中频增益
bandGains.put("high", 0.8);   // 高频增益
AudioData equalizedAudio = Audios.equalize(audioData, bandGains);

// 压缩器处理 / Compressor processing
AudioData compressedAudio = Audios.compress(audioData, 0.7, 4.0); // 阈值0.7，比率4:1
```


### 8. 音频嵌入 / Audio Embedding

#### 音频嵌入向量训练说明 / Audio Embedding Vector Training Instructions

音频嵌入向量训练是将音频信号转换为固定长度向量表示的过程，这些向量能够捕获音频的语义信息并用于音频检索、分类和相似性计算等任务。

Audio embedding vector training is the process of converting audio signals into fixed-length vector representations. These vectors can capture semantic information of audio and are used for tasks such as audio retrieval, classification, and similarity calculation.

##### i-vector模型训练 / i-vector Model Training

i-vector模型是一种常用的音频嵌入方法，它使用通用背景模型(UBM)和总变异性矩阵(T)来生成音频的紧凑向量表示。

The i-vector model is a commonly used audio embedding method that uses a Universal Background Model (UBM) and a Total Variability matrix (T) to generate compact vector representations of audio.

```java
import embedding.audio.com.yishape.lab.IVectorEmbedding;
import linalg.math.com.yishape.lab.IMatrix;

import java.util.List;

// 创建i-vector嵌入器 / Create i-vector embedder
IVectorEmbedding ivectorEmbedder = new IVectorEmbedding(64); // 64维i-vector

// 准备训练数据 / Prepare training data
List<IMatrix<Double>> trainingData = mfccList; // MFCC特征矩阵列表 / List of MFCC feature matrices

// 训练i-vector模型 / Train i-vector model
ivectorEmbedder.

train(trainingData);

// 使用训练好的模型提取嵌入向量 / Extract embedding vectors using the trained model
IVector<Double> embedding = ivectorEmbedder.embed(mfccMatrix);
```

##### 在线增量训练 / Online Incremental Training

对于大规模音频数据集，可以使用在线增量训练方法逐步更新模型参数，避免一次性加载所有数据到内存。

For large-scale audio datasets, online incremental training methods can be used to gradually update model parameters, avoiding loading all data into memory at once.

```java
import embedding.audio.com.yishape.lab.OnlineIVectorEmbedding;
import linalg.math.com.yishape.lab.IMatrix;

// 创建在线i-vector嵌入器 / Create online i-vector embedder
OnlineIVectorEmbedding onlineEmbedder = new OnlineIVectorEmbedding(64);

// 逐批次进行增量训练 / Perform incremental training batch by batch
for(
var filePath:paths){
IMatrix<Double> mfcc = Audios.readFile(filePath);
// 使用小批量MFCC样本进行增量训练 / Incremental training with small batch of MFCC samples
    onlineEmbedder.

trainIncremental(mfcc);
}

// 提取音频嵌入向量 / Extract audio embedding vectors
IVector<Double> embedding = onlineEmbedder.embed(audioData);
```

##### 训练参数配置 / Training Parameter Configuration

可以配置i-vector模型的各种参数以适应不同的应用场景。

Various parameters of the i-vector model can be configured to adapt to different application scenarios.

```java
// 创建具有自定义参数的i-vector嵌入器 / Create i-vector embedder with custom parameters
IVectorEmbedding customEmbedder = new IVectorEmbedding(
    64,   // i-vector维度 / i-vector dimension
    256,  // UBM高斯分量数 / Number of UBM Gaussian components
    13    // MFCC特征维度 / MFCC feature dimension
);

// 设置训练参数 / Set training parameters
Map<String, Object> parameters = new HashMap<>();
parameters.put("maxIterations", 100);
parameters.put("convergenceThreshold", 1e-6);
customEmbedder.setParameters(parameters);
```

##### 嵌入向量相似性计算 / Embedding Vector Similarity Calculation

训练完成后，可以计算不同音频嵌入向量之间的相似性。

After training, similarity between different audio embedding vectors can be calculated.

```java
// 计算两个嵌入向量之间的相似性 / Calculate similarity between two embedding vectors
double similarity = ivectorEmbedder.calculateSimilarity(embedding1, embedding2);

// 计算嵌入向量之间的距离 / Calculate distance between embedding vectors
double distance = ivectorEmbedder.calculateDistance(
    embedding1, 
    embedding2, 
    IAudioEmbedding.DistanceType.COSINE
);
```

##### 最佳实践 / Best Practices

1. **数据预处理** / **Data Preprocessing**
   - 确保训练数据具有代表性且覆盖目标应用领域
   - 对音频进行标准化处理以减少环境噪声影响
   - Ensure training data is representative and covers the target application domain
   - Standardize audio to reduce environmental noise impact

2. **参数调优** / **Parameter Tuning**
   - 根据数据集大小调整i-vector维度（通常32-128维）
   - 调整UBM高斯分量数以平衡模型复杂度和性能
   - Adjust i-vector dimension based on dataset size (typically 32-128 dimensions)
   - Tune UBM Gaussian component count to balance model complexity and performance

3. **内存管理** / **Memory Management**
   - 对于大规模数据集，使用在线增量训练方法
   - 及时释放不再需要的训练数据和中间结果
   - For large datasets, use online incremental training methods
   - Release training data and intermediate results that are no longer needed in time

4. **模型评估** / **Model Evaluation**
   - 使用验证集评估嵌入向量的质量
   - 通过相似性检索任务验证模型性能
   - Evaluate embedding vector quality using validation sets
   - Validate model performance through similarity retrieval tasks

## 数据结构 / Data Structures

### AudioData 类 / AudioData Class

```java
// 获取音频基本信息 / Get basic audio information
double sampleRate = audioData.getSampleRate();     // 采样率
int channels = audioData.getChannels();            // 声道数
int bitDepth = audioData.getBitDepth();            // 位深度
double duration = audioData.getDuration();         // 时长(秒)
AudioFormat format = audioData.getFormat();        // 音频格式

// 获取音频样本数据 / Get audio sample data
IVector<Double> samples = audioData.getSamples();  // 所有样本
IVector<Double> leftChannel = audioData.getChannel(0);  // 左声道
IVector<Double>[] allChannels = audioData.getAllChannels(); // 所有声道

// 音频数据操作 / Audio data operations
AudioData segment = audioData.extractSegment(1.0, 3.0);  // 提取1-3秒片段
AudioData normalized = audioData.normalize(1.0);         // 标准化
AudioData fadeIn = audioData.fadeIn(0.5);               // 淡入0.5秒
AudioData fadeOut = audioData.fadeOut(0.5);             // 淡出0.5秒
AudioData mixed = audioData.mixWith(otherAudio, 0.5);   // 混合音频

// 音频统计信息 / Audio statistics
double maxAmplitude = audioData.getMaxAmplitude();      // 最大幅度
double minAmplitude = audioData.getMinAmplitude();      // 最小幅度
double rmsAmplitude = audioData.getRMSAmplitude();      // RMS幅度
double avgAmplitude = audioData.getAverageAmplitude();  // 平均幅度
```

### AudioFormat 枚举 / AudioFormat Enum

```java
// 支持的音频格式 / Supported audio formats
AudioFormat.WAV     // WAV格式
AudioFormat.MP3     // MP3格式
AudioFormat.FLAC    // FLAC格式
AudioFormat.OGG     // OGG格式
AudioFormat.AAC     // AAC格式
AudioFormat.M4A     // M4A格式
AudioFormat.AIFF    // AIFF格式
AudioFormat.AU      // AU格式
AudioFormat.RAW     // RAW格式
```

## 工厂方法 / Factory Methods

### 组件创建 / Component Creation

```java
// 获取工厂实例 / Get factory instance
AudioComponentFactory factory = Audios.getFactory();

// 创建各种组件 / Create various components
IAdvancedAudioProcessor processor = factory.createProcessor("volume");
IAudioAnalyzer analyzer = factory.createAnalyzer("spectrum");
IBaseAudioFilter filter = factory.createFilter("lowpass");
IAudioEffect effect = factory.createEffect("reverb");
IAudioEnhancer enhancer = factory.createEnhancer("noise_reduction");
IAudioCodec codec = factory.createCodec("wav");
IAudioFeatureExtractor extractor = factory.createFeatureExtractor("standard");
```

## 异常处理 / Exception Handling

### 音频处理异常 / Audio Processing Exceptions

```java
import exception.audio.com.yishape.lab.AudioProcessingException;
import core.audio.com.yishape.lab.UnsupportedAudioFormatException;

try{
// 音频文件读取 / Audio file reading
AudioData audio = AudioIO.readAudio("path/to/audio.wav");

// 音频处理 / Audio processing
AudioData processed = Audios.adjustVolume(audio, 1.5);

// 音频文件写入 / Audio file writing
    AudioIO.

writeAudio(processed, "output/processed.wav");
    
}catch(
UnsupportedAudioFormatException e){
        System.err.

println("不支持的音频格式: "+e.getMessage());
        }catch(
AudioProcessingException e){
        System.err.

println("音频处理错误: "+e.getMessage());
        }catch(
IOException e){
        System.err.

println("文件操作错误: "+e.getMessage());
        }
```

## 性能优化建议 / Performance Optimization Tips

### 1. 内存管理 / Memory Management
- 处理大音频文件时，考虑分段处理以减少内存占用
- 及时释放不再使用的音频数据对象

### 2. 并行处理 / Parallel Processing
- 对于多声道音频，可以并行处理各个声道
- 使用批处理模式处理多个音频文件

### 3. 缓存策略 / Caching Strategy
- 对于重复使用的特征提取结果，考虑缓存
- 预计算常用的滤波器系数

## 最佳实践 / Best Practices

### 1. 错误处理 / Error Handling
- 始终使用try-catch块处理音频操作
- 验证音频数据的有效性

### 2. 资源管理 / Resource Management
- 使用try-with-resources处理文件操作
- 及时关闭音频流和资源

### 3. 参数验证 / Parameter Validation
- 验证音频处理参数的合理性
- 检查音频格式兼容性

### 4. 性能监控 / Performance Monitoring
- 监控音频处理的执行时间
- 记录内存使用情况
