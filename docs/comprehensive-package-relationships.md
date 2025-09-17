# 综合包关系图 / Comprehensive Package Relationships

## 包间关系概览 / Package Relationship Overview

```mermaid
graph TB
    subgraph "核心数学库 / Core Math Library"
        A[IVector] --> B[IMatrix]
        C[Linalg] --> A
        C --> B
        D[Stats] --> E[Probability Distributions]
    end
    
    subgraph "信号处理包 / Signal Processing Package"
        F[SignalGeneration] --> G[SignalAnalysis]
        F --> H[SignalFiltering]
        F --> I[SignalUtilities]
        J[RereFFT] --> G
        K[RereDCT] --> G
        L[Complex] --> J
        M[WaveletAnalysis] --> G
        N[WaveletFilters] --> M
    end
    
    subgraph "音频处理包 / Audio Processing Package"
        O[AudioData] --> P[AudioProcessor]
        O --> Q[AudioAnalyzer]
        O --> R[AudioVisualizer]
        O --> S[AudioStatistics]
        O --> T[AudioFeatures]
        U[MusicTheory] --> V[MusicAnalyzer]
        U --> W[MusicGenerator]
        U --> X[MusicVisualizer]
    end
    
    subgraph "图像处理包 / Image Processing Package"
        Y[ImageData] --> Z[ImageTransform]
        Y --> AA[ImageFilter]
        Y --> BB[ImageMorphology]
        Y --> CC[ImageSegmentation]
        Y --> DD[ImageFeatures]
        Y --> EE[ImageUtils]
    end
    
    subgraph "可视化包 / Visualization Package"
        FF[Plots] --> GG[IPlot]
        HH[ThemeManager] --> FF
        II[ColorPalette] --> FF
    end
    
    subgraph "机器学习包 / Machine Learning Package"
        JJ[Classification] --> KK[Regression]
        LL[Optimization] --> JJ
        LL --> KK
    end
    
    subgraph "数据包 / Data Package"
        MM[DataFrame] --> NN[Column]
        OO[ColumnType] --> NN
    end
    
    %% 跨包依赖关系 / Cross-package Dependencies
    A --> O
    B --> Q
    A --> Y
    B --> Z
    A --> F
    B --> G
    
    G --> Q
    H --> P
    I --> P
    
    FF --> R
    FF --> X
    FF --> Z
    
    D --> S
    E --> S
    
    JJ --> T
    KK --> T
    
    MM --> O
    MM --> Y
    MM --> F
```

## 详细类关系图 / Detailed Class Relationship Diagram

```mermaid
classDiagram
    class IVector {
        <<interface>>
        +length() int
        +get(int) Double
        +set(int, Double) IVector
        +add(IVector) IVector
        +multiply(IVector) IVector
        +mean() Double
        +std() Double
    }
    
    class IMatrix {
        <<interface>>
        +getRowNum() int
        +getColNum() int
        +get(int, int) Double
        +set(int, int, Double) IMatrix
        +multiply(IMatrix) IMatrix
        +transpose() IMatrix
    }
    
    class AudioData {
        -samples: IVector~Double~
        -sampleRate: double
        -channels: int
        -bitDepth: int
        -duration: double
        -format: AudioFormat
        +getSamples() IVector~Double~
        +getChannel(int) IVector~Double~
        +isMono() boolean
        +isStereo() boolean
    }
    
    class AudioAnalyzer {
        +calculateSpectrum(AudioData) Tuple2
        +calculateSTFT(AudioData, int, int) IMatrix~Double~
        +extractFeatures(AudioData) AudioFeatures
        +detectPitch(AudioData) double
    }
    
    class AudioProcessor {
        +adjustVolume(AudioData, double) AudioData
        +normalize(AudioData) AudioData
        +monoToStereo(AudioData) AudioData
        +stereoToMono(AudioData) AudioData
        +resample(AudioData, double) AudioData
        +filter(AudioData, FilterType, double) AudioData
    }
    
    class MusicTheory {
        +generateScale(int, ScaleType) int[]
        +generateChord(int, ChordType) int[]
        +detectKey(AudioData) Key
        +detectChord(AudioData) Chord
        +semitonesToFrequency(int) double
        +frequencyToSemitones(double) int
    }
    
    class MusicAnalyzer {
        +detectBeats(AudioData) BeatDetectionResult
        +extractMusicFeatures(AudioData) MusicFeatures
        +analyzeMusicStructure(AudioData) String
        +detectMusicGenre(AudioData) String
        +calculateMusicSimilarity(AudioData, AudioData) double
    }
    
    class MusicGenerator {
        +generateRandomMelody(int, ScaleType, int, int, double, int) Melody
        +generateChordProgression(Key, ChordType[], int, double, double) AudioData
        +generateScaleExercise(int, ScaleType, int, double, double) AudioData
        +generateMetronome(double, double, double, int) AudioData
    }
    
    class ImageData {
        -width: int
        -height: int
        -channels: int
        -data: IVector~Double~
        -format: ImageFormat
        +getPixel(int, int, int) double
        +setPixel(int, int, int, double) void
        +getChannel(int) IVector~Double~
        +separateChannels() ImageData[]
    }
    
    class ImageTransform {
        +translate(ImageData, int, int) ImageData
        +rotate(ImageData, double) ImageData
        +scale(ImageData, double, double) ImageData
        +fft2D(ImageData) IMatrix~Complex~
        +dct2D(ImageData) IMatrix~Double~
    }
    
    class ImageFilter {
        +gaussianFilter(ImageData, double) ImageData
        +laplacianFilter(ImageData) ImageData
        +sobelEdgeDetection(ImageData) ImageData
        +medianFilter(ImageData, int) ImageData
        +bilateralFilter(ImageData, double, double) ImageData
    }
    
    class ImageMorphology {
        +erode(ImageData, IMatrix~Double~) ImageData
        +dilate(ImageData, IMatrix~Double~) ImageData
        +opening(ImageData, IMatrix~Double~) ImageData
        +closing(ImageData, IMatrix~Double~) ImageData
    }
    
    class SignalGeneration {
        +sineWave(int, double, double, double, double) IVector~Double~
        +cosineWave(int, double, double, double, double) IVector~Double~
        +squareWave(int, double, double, double, double) IVector~Double~
        +whiteNoise(int, double) IVector~Double~
        +pinkNoise(int, double) IVector~Double~
    }
    
    class SignalAnalysis {
        +powerSpectralDensity(IVector~Double~, int, double, double) Tuple2
        +shortTimeFourierTransform(IVector~Double~, int, int) IMatrix~Double~
        +autocorrelation(IVector~Double~) IVector~Double~
        +crossCorrelation(IVector~Double~, IVector~Double~) IVector~Double~
        +signalToNoiseRatio(IVector~Double~, IVector~Double~) double
    }
    
    class SignalFiltering {
        +movingAverage(IVector~Double~, int) IVector~Double~
        +medianFilter(IVector~Double~, int) IVector~Double~
        +gaussianFilter(IVector~Double~, double) IVector~Double~
        +butterworthLowPass(IVector~Double~, double, double, int) IVector~Double~
        +butterworthHighPass(IVector~Double~, double, double, int) IVector~Double~
        +bandPass(IVector~Double~, double, double, double, int) IVector~Double~
    }
    
    class RereFFT {
        +fft(Complex[]) Complex[]
        +ifft(Complex[]) Complex[]
        +magnitudeSpectrum(Complex[]) IVector~Double~
        +phaseSpectrum(Complex[]) IVector~Double~
        +powerSpectrum(Complex[]) IVector~Double~
    }
    
    class WaveletAnalysis {
        +continuousWaveletTransform(IVector~Double~, double[], String) IMatrix~Double~
        +discreteWaveletTransform(IVector~Double~, String, int) Tuple2
        +softThresholding(IVector~Double~, double) IVector~Double~
        +hardThresholding(IVector~Double~, double) IVector~Double~
    }
    
    %% 关系定义 / Relationship Definitions
    AudioData --> IVector : uses
    AudioAnalyzer --> IMatrix : uses
    AudioProcessor --> SignalFiltering : uses
    MusicTheory --> SignalGeneration : uses
    MusicAnalyzer --> AudioAnalyzer : uses
    MusicGenerator --> MusicTheory : uses
    
    ImageData --> IVector : uses
    ImageTransform --> IMatrix : uses
    ImageFilter --> SignalFiltering : uses
    
    SignalAnalysis --> RereFFT : uses
    SignalFiltering --> SignalGeneration : uses
    WaveletAnalysis --> SignalAnalysis : uses
```

## 数据流图 / Data Flow Diagram

```mermaid
flowchart TD
    A[原始数据 / Raw Data] --> B{数据类型 / Data Type}
    
    B -->|音频 / Audio| C[AudioData]
    B -->|图像 / Image| D[ImageData]
    B -->|信号 / Signal| E[IVector~Double~]
    
    C --> F[AudioProcessor]
    C --> G[AudioAnalyzer]
    C --> H[MusicTheory]
    
    D --> I[ImageTransform]
    D --> J[ImageFilter]
    D --> K[ImageMorphology]
    
    E --> L[SignalGeneration]
    E --> M[SignalAnalysis]
    E --> N[SignalFiltering]
    
    F --> O[处理后的音频 / Processed Audio]
    G --> P[音频特征 / Audio Features]
    H --> Q[音乐特征 / Music Features]
    
    I --> R[变换后的图像 / Transformed Image]
    J --> S[滤波后的图像 / Filtered Image]
    K --> T[形态学处理后的图像 / Morphologically Processed Image]
    
    L --> U[生成的信号 / Generated Signal]
    M --> V[分析结果 / Analysis Results]
    N --> W[滤波后的信号 / Filtered Signal]
    
    O --> X[可视化 / Visualization]
    P --> X
    Q --> X
    R --> X
    S --> X
    T --> X
    U --> X
    V --> X
    W --> X
    
    X --> Y[最终输出 / Final Output]
```

## 功能层次图 / Functionality Hierarchy

```mermaid
graph TD
    A[数学计算库 / Math Library] --> B[线性代数 / Linear Algebra]
    A --> C[统计学 / Statistics]
    A --> D[优化算法 / Optimization]
    
    B --> E[向量运算 / Vector Operations]
    B --> F[矩阵运算 / Matrix Operations]
    
    C --> G[概率分布 / Probability Distributions]
    C --> H[假设检验 / Hypothesis Testing]
    
    E --> I[信号处理 / Signal Processing]
    F --> I
    E --> J[音频处理 / Audio Processing]
    F --> J
    E --> K[图像处理 / Image Processing]
    F --> K
    
    I --> L[信号生成 / Signal Generation]
    I --> M[信号分析 / Signal Analysis]
    I --> N[信号滤波 / Signal Filtering]
    I --> O[小波分析 / Wavelet Analysis]
    
    J --> P[音频数据 / Audio Data]
    J --> Q[音频处理 / Audio Processing]
    J --> R[音频分析 / Audio Analysis]
    J --> S[音乐理论 / Music Theory]
    J --> T[音乐分析 / Music Analysis]
    J --> U[音乐生成 / Music Generation]
    
    K --> V[图像数据 / Image Data]
    K --> W[图像变换 / Image Transform]
    K --> X[图像滤波 / Image Filter]
    K --> Y[图像形态学 / Image Morphology]
    K --> Z[图像分割 / Image Segmentation]
    K --> AA[图像特征 / Image Features]
    
    L --> BB[可视化 / Visualization]
    M --> BB
    N --> BB
    O --> BB
    P --> BB
    Q --> BB
    R --> BB
    S --> BB
    T --> BB
    U --> BB
    V --> BB
    W --> BB
    X --> BB
    Y --> BB
    Z --> BB
    AA --> BB
```

## 使用场景图 / Usage Scenarios

```mermaid
graph LR
    A[用户输入 / User Input] --> B{应用类型 / Application Type}
    
    B -->|音频应用 / Audio App| C[音频处理流程 / Audio Processing Flow]
    B -->|图像应用 / Image App| D[图像处理流程 / Image Processing Flow]
    B -->|信号应用 / Signal App| E[信号处理流程 / Signal Processing Flow]
    
    C --> C1[加载音频 / Load Audio]
    C1 --> C2[音频预处理 / Audio Preprocessing]
    C2 --> C3[特征提取 / Feature Extraction]
    C3 --> C4[音乐分析 / Music Analysis]
    C4 --> C5[结果可视化 / Result Visualization]
    
    D --> D1[加载图像 / Load Image]
    D1 --> D2[图像预处理 / Image Preprocessing]
    D2 --> D3[图像变换 / Image Transform]
    D3 --> D4[特征提取 / Feature Extraction]
    D4 --> D5[结果可视化 / Result Visualization]
    
    E --> E1[生成信号 / Generate Signal]
    E1 --> E2[信号滤波 / Signal Filtering]
    E2 --> E3[频谱分析 / Spectral Analysis]
    E3 --> E4[特征提取 / Feature Extraction]
    E4 --> E5[结果可视化 / Result Visualization]
    
    C5 --> F[输出结果 / Output Results]
    D5 --> F
    E5 --> F
```

## 性能优化关系图 / Performance Optimization Relationships

```mermaid
graph TD
    A[性能优化 / Performance Optimization] --> B[内存管理 / Memory Management]
    A --> C[计算优化 / Computational Optimization]
    A --> D[并行处理 / Parallel Processing]
    
    B --> B1[流式处理 / Streaming Processing]
    B --> B2[内存池 / Memory Pool]
    B --> B3[垃圾回收优化 / GC Optimization]
    
    C --> C1[FFT优化 / FFT Optimization]
    C --> C2[向量化操作 / Vectorized Operations]
    C --> C3[算法优化 / Algorithm Optimization]
    
    D --> D1[多线程处理 / Multi-threading]
    D --> D2[GPU加速 / GPU Acceleration]
    D --> D3[分布式处理 / Distributed Processing]
    
    B1 --> E[音频流处理 / Audio Stream Processing]
    B1 --> F[图像块处理 / Image Tile Processing]
    B1 --> G[信号块处理 / Signal Chunk Processing]
    
    C1 --> H[快速卷积 / Fast Convolution]
    C1 --> I[快速相关 / Fast Correlation]
    
    C2 --> J[矩阵运算优化 / Matrix Operations Optimization]
    C2 --> K[向量运算优化 / Vector Operations Optimization]
    
    D1 --> L[并行FFT / Parallel FFT]
    D1 --> M[并行滤波 / Parallel Filtering]
    D1 --> N[并行特征提取 / Parallel Feature Extraction]
```

这个综合关系图展示了三个包之间的复杂关系，包括：

1. **依赖关系**：哪些类依赖于其他类
2. **数据流**：数据如何在不同的处理阶段之间流动
3. **功能层次**：从基础数学库到高级应用功能的层次结构
4. **使用场景**：不同应用类型如何使用这些包
5. **性能优化**：如何优化整个系统的性能

这些关系图帮助理解整个系统的架构，以及如何有效地使用和扩展这些包。
