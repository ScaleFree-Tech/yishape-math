```
src/main/java/
└── com/yishape/lab/
    ├── audio/                    # 音频处理模块 / Audio Processing Module
    │   ├── AudioPlots.java       # 音频绘图 / Audio Plots
    │   ├── Audios.java           # 音频处理入口类 / Audio Processing Entry Class
    │   ├── analysis/             # 音频分析 / Audio Analysis
    │   │   ├── AbstractAudioAnalyzer.java  # 抽象音频分析器 / Abstract Audio Analyzer
    │   │   ├── FFTProcessor.java           # FFT处理器 / FFT Processor
    │   │   ├── IAudioAnalyzer.java         # 音频分析接口 / Audio Analyzer Interface
    │   │   ├── PitchDetector.java          # 音高检测器 / Pitch Detector
    │   │   ├── STFTAnalyzer.java           # 短时傅里叶变换分析器 / STFT Analyzer
    │   │   └── SpectrumAnalyzer.java       # 频谱分析器 / Spectrum Analyzer
    │   ├── core/                 # 核心类 / Core Classes
    │   │   ├── AudioData.java              # 音频数据类 / Audio Data Class
    │   │   ├── AudioFormat.java            # 音频格式类 / Audio Format Class
    │   │   ├── AudioIO.java                # 音频输入输出 / Audio I/O
    │   │   ├── AudioProcessor.java         # 音频处理器 / Audio Processor
    │   │   ├── AudioQuality.java           # 音频质量类 / Audio Quality Class
    │   │   ├── AudioStatistics.java        # 音频统计类 / Audio Statistics Class
    │   │   ├── AudioUtil.java              # 音频工具类 / Audio Utilities Class
    │   │   ├── CompressionSettings.java    # 压缩设置 / Compression Settings
    │   │   ├── EqualizerBand.java          # 均衡器频段 / Equalizer Band
    │   │   ├── EqualizerSettings.java      # 均衡器设置 / Equalizer Settings
    │   │   ├── IAudioCodec.java            # 音频编解码接口 / Audio Codec Interface
    │   │   ├── IAudioComponentStandard.java # 音频组件标准接口 / Audio Component Standard Interface
    │   │   ├── IAudioListener.java         # 音频监听器接口 / Audio Listener Interface
    │   │   ├── NoiseProfile.java           # 噪声配置文件 / Noise Profile
    │   │   ├── ReverbSettings.java         # 混响设置 / Reverb Settings
    │   │   └── UnsupportedAudioFormatException.java # 不支持的音频格式异常 / Unsupported Audio Format Exception
    │   ├── effect/               # 音频效果 / Audio Effects
    │   │   ├── AbstractAudioEffect.java    # 抽象音频效果 / Abstract Audio Effect
    │   │   ├── IAudioEffect.java           # 音频效果接口 / Audio Effect Interface
    │   │   └── ReverbEffect.java           # 混响效果 / Reverb Effect
    │   ├── embedding/            # 音频嵌入 / Audio Embedding
    │   │   ├── IAudioEmbedding.java        # 音频嵌入接口 / Audio Embedding Interface
    │   │   ├── IVectorEmbedding.java       # i-Vector语音向量嵌入 / i-Vector Audio Embedding
    │   │   └── OnlineIVectorEmbedding.java # i-Vector语音向量嵌入 / Online i-Vector Audio Embedding
    │   ├── enhancement/          # 音频增强 / Audio Enhancement
    │   │   ├── AbstractAudioEnhancer.java  # 抽象音频增强器 / Abstract Audio Enhancer
    │   │   ├── CompressorEnhancer.java     # 压缩器增强 / Compressor Enhancer
    │   │   ├── EqualizerEnhancer.java      # 均衡器增强 / Equalizer Enhancer
    │   │   ├── IAudioEnhancer.java         # 音频增强接口 / Audio Enhancer Interface
    │   │   ├── NoiseReductionEnhancer.java # 噪声抑制增强 / Noise Reduction Enhancer
    │   │   └── ReverbEnhancer.java         # 混响增强 / Reverb Enhancer
    │   ├── exception/            # 音频异常 / Audio Exceptions
    │   │   ├── AudioAnalysisException.java # 音频分析异常 / Audio Analysis Exception
    │   │   └── AudioProcessingException.java # 音频处理异常 / Audio Processing Exception
    │   ├── factory/              # 音频工厂 / Audio Factory
    │   │   └── AudioComponentFactory.java  # 音频组件工厂 / Audio Component Factory
    │   ├── feature/              # 音频特征 / Audio Features
    │   │   ├── AudioFeatureExtractor.java  # 音频特征提取器 / Audio Feature Extractor
    │   │   ├── AudioFeatureExtractorImpl.java # 音频特征提取器实现 / Audio Feature Extractor Implementation
    │   │   ├── AudioFeatureResult.java     # 音频特征结果 / Audio Feature Result
    │   │   ├── FrequencyDomainFeatureResult.java # 频域特征结果 / Frequency Domain Feature Result
    │   │   ├── IAudioFeatureExtractor.java # 音频特征提取接口 / Audio Feature Extractor Interface
    │   │   ├── SpectralFeatureResult.java  # 谱特征结果 / Spectral Feature Result
    │   │   └── TimeDomainFeatureResult.java # 时域特征结果 / Time Domain Feature Result
    │   ├── filter/               # 音频滤波器 / Audio Filters
    │   │   ├── AbstractAdvancedAudioFilter.java # 抽象高级音频滤波器 / Abstract Advanced Audio Filter
    │   │   ├── AbstractAudioFilterStandard.java # 抽象音频滤波器标准 / Abstract Audio Filter Standard
    │   │   ├── AdvancedLowPassFilter.java  # 高级低通滤波器 / Advanced Low Pass Filter
    │   │   ├── IAdvancedAudioFilter.java   # 高级音频滤波器接口 / Advanced Audio Filter Interface
    │   │   ├── IBaseAudioFilter.java       # 基础音频滤波器接口 / Base Audio Filter Interface
    │   │   └── LowPassFilter.java          # 低通滤波器 / Low Pass Filter
    │   ├── generation/           # 音频生成 / Audio Generation
    │   ├── pipeline/             # 音频处理管道 / Audio Pipeline
    │   │   └── AudioPipelineBuilder.java   # 音频管道构建器 / Audio Pipeline Builder
    │   ├── preprocessing/        # 音频预处理 / Audio Preprocessing
    │   │   ├── AudioPreprocessingOptions.java # 音频预处理选项 / Audio Preprocessing Options
    │   │   └── AudioPreprocessor.java      # 音频预处理器 / Audio Preprocessor
    │   ├── processing/           # 音频处理 / Audio Processing
    │   │   ├── AbstractAudioProcessorStandard.java # 抽象音频处理器标准 / Abstract Audio Processor Standard
    │   │   ├── ChannelProcessor.java       # 通道处理器 / Channel Processor
    │   │   ├── IAdvancedAudioProcessor.java # 高级音频处理器接口 / Advanced Audio Processor Interface
    │   │   ├── IAudioProcessor.java        # 音频处理器接口 / Audio Processor Interface
    │   │   ├── IBaseAudioProcessor.java    # 基础音频处理器接口 / Base Audio Processor Interface
    │   │   ├── NormalizeProcessor.java     # 归一化处理器 / Normalize Processor
    │   │   └── VolumeProcessor.java        # 音量处理器 / Volume Processor
    │   └── transform/            # 音频变换 / Audio Transform
    ├── image/                    # 图像处理模块 / Image Processing Module
    │   ├── ImageData.java        # 图像数据类 / Image Data Class
    │   ├── ImageFeatures.java    # 图像特征 / Image Features
    │   ├── ImageFilter.java      # 图像滤波器 / Image Filter
    │   ├── ImageMorphology.java  # 图像形态学 / Image Morphology
    │   ├── ImageSegmentation.java # 图像分割 / Image Segmentation
    │   ├── ImageTransform.java   # 图像变换 / Image Transform
    │   ├── ImageUtils.java       # 图像工具类 / Image Utilities
    │   ├── core/                 # 核心类 / Core Classes
    │   │   ├── IImageAnalyzer.java     # 图像分析接口 / Image Analyzer Interface
    │   │   ├── IImageFilter.java       # 图像滤波接口 / Image Filter Interface
    │   │   ├── IImageProcessor.java    # 图像处理器接口 / Image Processor Interface
    │   │   ├── IImageSegmenter.java    # 图像分割接口 / Image Segmenter Interface
    │   │   ├── IImageTransformer.java  # 图像变换接口 / Image Transformer Interface
    │   │   └── ImageProcessingException.java # 图像处理异常 / Image Processing Exception
    │   ├── detection/            # 检测 / Detection
    │   │   └── AdvancedFeatureDetection.java # 高级特征检测 / Advanced Feature Detection
    │   ├── enhancement/          # 增强 / Enhancement
    │   │   └── AdvancedImageEnhancement.java # 高级图像增强 / Advanced Image Enhancement
    │   ├── factory/              # 工厂 / Factory
    │   │   ├── ImageComponentFactory.java # 图像组件工厂 / Image Component Factory
    │   │   └── ImagePipelineBuilder.java  # 图像管道构建器 / Image Pipeline Builder
    │   ├── features/             # 特征 / Features
    │   │   ├── AdvancedImageFeatures.java # 高级图像特征 / Advanced Image Features
    │   │   ├── SIFTFeatureDetector.java   # SIFT特征检测器 / SIFT Feature Detector
    │   │   └── SURFFeatureDetector.java   # SURF特征检测器 / SURF Feature Detector
    │   └── tracking/             # 跟踪 / Tracking
    │       └── OpticalFlowTracker.java    # 光流跟踪器 / Optical Flow Tracker
    ├── math/                     # 数学计算模块 / Mathematical Computing Module
    │   ├── RereMathUtil.java     # 数学工具类 / Math Utilities Class
    │   ├── compute/              # 计算模块 / Computing Module
    │   │   ├── ComputerConfig.java       # 计算配置 / Computer Config
    │   │   ├── DoubleVectorComputer.java # 双精度向量计算机 / Double Vector Computer
    │   │   ├── FloatVectorComputer.java  # 单精度向量计算机 / Float Vector Computer
    │   │   ├── IDoubleVectorComputer.java # 双精度向量计算机接口 / Double Vector Computer Interface
    │   │   ├── IFloatVectorComputer.java  # 单精度向量计算机接口 / Float Vector Computer Interface
    │   │   ├── SIMDDoubleComputer.java   # SIMD双精度计算机 / SIMD Double Computer
    │   │   ├── SIMDFloatComputer.java    # SIMD单精度计算机 / SIMD Float Computer
    │   │   ├── SISDDoubleComputer.java   # SISD双精度计算机 / SISD Double Computer
    │   │   └── SISDFloatComputer.java    # SISD单精度计算机 / SISD Float Computer
    │   ├── data/                 # 数据结构模块 / Data Structure Module
    │   │   ├── Column.java       # 列类 / Column Class
    │   │   ├── ColumnType.java   # 列类型枚举 / Column Type Enum
    │   │   └── DataFrame.java    # 数据框类 / DataFrame Class
    │   ├── linalg/               # 线性代数模块 / Linear Algebra Module
    │   │   ├── IDoubleMatrix.java    # 双精度矩阵接口 / Double Matrix Interface
    │   │   ├── IDoubleVector.java    # 双精度向量接口 / Double Vector Interface
    │   │   ├── IFloatMatrix.java     # 单精度矩阵接口 / Float Matrix Interface
    │   │   ├── IFloatVector.java     # 单精度向量接口 / Float Vector Interface
    │   │   ├── IMatrix.java          # 矩阵操作接口 / Matrix Operations Interface
    │   │   ├── IVector.java          # 向量操作接口 / Vector Operations Interface
    │   │   ├── Linalg.java           # 线性代数工具类 / Linear Algebra Utilities
    │   │   ├── RereDoubleMatrix.java # 双精度矩阵实现类 / Double Matrix Implementation
    │   │   ├── RereDoubleVector.java # 双精度向量实现类 / Double Vector Implementation
    │   │   ├── RereFloatMatrix.java  # 单精度矩阵实现类 / Float Matrix Implementation
    │   │   ├── RereFloatVector.java  # 单精度向量实现类 / Float Vector Implementation
    │   │   ├── SliceExpressionParser.java # 切片表达式解析器 / Slice Expression Parser
    │   │   ├── decomposition/        # 矩阵分解 / Matrix Decomposition
    │   │   │   ├── BlockOperationUtils.java      # 块操作工具 / Block Operation Utils
    │   │   │   ├── DecompositionFailedException.java # 分解失败异常 / Decomposition Failed Exception
    │   │   │   ├── Decomps.java                  # 分解工具 / Decomposition Utils
    │   │   │   ├── IBidiagonalDecomposition.java # 双对角分解接口 / Bidiagonal Decomposition Interface
    │   │   │   ├── ICholeskyDecomposition.java   # Cholesky分解接口 / Cholesky Decomposition Interface
    │   │   │   ├── IEigenDecomposition.java      # 特征分解接口 / Eigen Decomposition Interface
    │   │   │   ├── IHessenbergDecomposition.java # Hessenberg分解接口 / Hessenberg Decomposition Interface
    │   │   │   ├── ILUDecomposition.java         # LU分解接口 / LU Decomposition Interface
    │   │   │   ├── IMatrixDecomposition.java     # 矩阵分解接口 / Matrix Decomposition Interface
    │   │   │   ├── IQRDecomposition.java         # QR分解接口 / QR Decomposition Interface
    │   │   │   ├── ISchurDecomposition.java      # Schur分解接口 / Schur Decomposition Interface
    │   │   │   ├── ISVDDecomposition.java        # SVD分解接口 / SVD Decomposition Interface
    │   │   │   ├── ITridiagonalDecomposition.java # 三对角分解接口 / Tridiagonal Decomposition Interface
    │   │   │   ├── MatrixDecompositionException.java # 矩阵分解异常 / Matrix Decomposition Exception
    │   │   │   ├── NonPositiveDefiniteMatrixException.java # 非正定矩阵异常 / Non Positive Definite Matrix Exception
    │   │   │   ├── NonSquareMatrixException.java # 非方阵异常 / Non Square Matrix Exception
    │   │   │   ├── NonSymmetricMatrixException.java # 非对称矩阵异常 / Non Symmetric Matrix Exception
    │   │   │   ├── SingularMatrixException.java  # 奇异矩阵异常 / Singular Matrix Exception
    │   │   └── solver/                # 求解器 / Solver
    │   │       ├── ConditionNumberSolver.java   # 条件数求解器 / Condition Number Solver
    │   │       ├── DeterminantSolver.java       # 行列式求解器 / Determinant Solver
    │   │       ├── LeastSquaresSolver.java      # 最小二乘求解器 / Least Squares Solver
    │   │       ├── LinearSystemSolver.java      # 线性系统求解器 / Linear System Solver
    │   │       ├── MatrixInversionSolver.java   # 矩阵求逆求解器 / Matrix Inversion Solver
    │   │       ├── RankSolver.java              # 秩求解器 / Rank Solver
    │   ├── ml/                   # 机器学习算法 / Machine Learning Algorithms
    │   │   ├── ISerializableModel.java # 可序列化模型接口 / Serializable Model Interface
    │   │   ├── MLModelType.java       # 机器学习模型类型 / ML Model Type
    │   │   ├── cls/              # 分类算法 / Classification Algorithms
    │   │   │   ├── ClassificationResult.java    # 分类结果 / Classification Result
    │   │   │   ├── EnsembleClassifier.java      # 集成分类器 / Ensemble Classifier
    │   │   │   ├── EnsembleResult.java          # 集成结果 / Ensemble Result
    │   │   │   ├── IClassification.java         # 分类接口 / Classification Interface
    │   │   │   ├── LogisticRegressionResult.java # 逻辑回归结果 / Logistic Regression Result
    │   │   │   ├── RereLogisticRegression.java  # 逻辑回归实现 / Logistic Regression Implementation
    │   │   ├── clustering/       # 聚类算法 / Clustering Algorithms
    │   │   │   ├── ClusteringMetrics.java       # 聚类评估指标 / Clustering Metrics
    │   │   │   ├── GMMClustering.java           # 高斯混合模型聚类 / GMM Clustering
    │   │   │   ├── IClustering.java             # 聚类接口 / Clustering Interface
    │   │   │   ├── KMeansPlusPlus.java          # K-Means++聚类 / K-Means++ Clustering
    │   │   ├── dimreduce/        # 降维算法 / Dimensionality Reduction Algorithms
    │   │   │   ├── IDimReduce.java              # 降维接口 / Dimensionality Reduction Interface
    │   │   │   ├── RerePCA.java                 # PCA降维 / PCA Dimensionality Reduction
    │   │   │   ├── RereSVD.java                 # SVD降维 / SVD Dimensionality Reduction
    │   │   │   ├── RereTSNE.java                # t-SNE降维 / t-SNE Dimensionality Reduction
    │   │   │   └── RereUMAP.java                # UMAP降维 / UMAP Dimensionality Reduction
    │   │   ├── lr/               # 线性回归 / Linear Regression
    │   │   │   ├── IRegression.java             # 回归接口 / Regression Interface
    │   │   │   ├── RegressionResult.java        # 回归结果 / Regression Result
    │   │   │   ├── RereLinearRegression.java    # 线性回归实现 / Linear Regression Implementation
    │   │   └── metric/            # 评估指标 / Metrics
    │   │       ├── ClassificationMetrics.java   # 分类评估指标 / Classification Metrics
    │   │       ├── ClassificationMetricsExample.java # 分类评估指标示例 / Classification Metrics Example
    │   │       ├── CrossValidation.java         # 交叉验证 / Cross Validation
    │   │       ├── MetricsDemo.java             # 评估指标演示 / Metrics Demo
    │   ├── optimize/             # 优化算法 / Optimization Algorithms
    │   │   ├── IGradientFunction.java  # 梯度函数接口 / Gradient Function Interface
    │   │   ├── IObjectiveFunction.java # 目标函数接口 / Objective Function Interface
    │   │   ├── IOnlineOptimizer.java  # 在线优化器接口 / Online Optimizer Interface
    │   │   ├── IOptimizer.java       # 优化器接口 / Optimizer Interface
    │   │   ├── OptResult.java        # 优化结果 / Optimization Result
    │   │   ├── OptimizerExample.java # 优化器示例 / Optimizer Example
    │   │   ├── Opts.java             # 优化工具类 / Optimization Utilities
    │   │   ├── RereLineSearch.java   # 线搜索 / Line Search
    │   │   ├── constraint/           # 约束优化 / Constraint Optimization
    │   │   │   └── LagrangeMultiplierSolver.java # 拉格朗日乘数求解器 / Lagrange Multiplier Solver
    │   │   ├── linpg/                # 线性规划 / Linear Programming
    │   │   │   ├── IIntegerProg.java           # 整数规划接口 / Integer Programming Interface
    │   │   │   ├── ILinProgSolver.java         # 线性规划求解器接口 / Linear Programming Solver Interface
    │   │   │   ├── InteriorPointLinProgSolver.java # 内点法线性规划求解器 / Interior Point Linear Programming Solver
    │   │   │   ├── LangMultiplierLinProgSolver.java # 拉格朗日乘数线性规划求解器 / Lagrange Multiplier Linear Programming Solver
    │   │   │   ├── LinProgUtil.java            # 线性规划工具类 / Linear Programming Utilities
    │   │   │   ├── RereIntegerProg.java        # 整数规划实现 / Integer Programming Implementation
    │   │   │   └── simplex/             # 单纯形法 / Simplex Method
    │   │   │       ├── ConstraintType.java      # 约束类型 / Constraint Type
    │   │   │       ├── ILinearConstraint.java   # 线性约束接口 / Linear Constraint Interface
    │   │   │       ├── ILinearConstraint.java   # 线性约束 / Linear Constraint
    │   │   │       ├── ISimplexLinProgSolver.java # 单纯形线性规划求解器接口 / Simplex Linear Programming Solver Interface
    │   │   │       ├── LinearConstraint.java    # 线性约束 / Linear Constraint
    │   │   │       ├── PivotSelectionRule.java  # 主元选择规则 / Pivot Selection Rule
    │   │   │       ├── PivotSelectionStrategy.java # 主元选择策略 / Pivot Selection Strategy
    │   │   │       ├── RereSimplexLinProgSolver.java # 单纯形线性规划求解器 / Simplex Linear Programming Solver
    │   │   │       ├── RereSimplexTableau.java  # 单纯形表 / Simplex Tableau
    │   │   └── newton/               # 牛顿法优化 / Newton Method Optimization
    │   │       ├── RereConjugateGradient.java  # 共轭梯度法 / Conjugate Gradient Method
    │   │       ├── RereDFP.java                # DFP算法 / DFP Algorithm
    │   │       ├── RereLBFGS.java              # L-BFGS优化器 / L-BFGS Optimizer
    │   │       ├── RereOnlineAdam.java         # 在线Adam优化器 / Online Adam Optimizer
    │   │       ├── RereOnlineSGD.java          # 在线SGD优化器 / Online SGD Optimizer
    │   │       └── RereSteepestDescent.java    # 最速下降法 / Steepest Descent Method
    │   ├── signal/               # 信号处理模块 / Signal Processing Module
    │   │   ├── SignalPlots.java      # 信号绘图 / Signal Plots
    │   │   ├── Signals.java          # 信号处理工具类 / Signal Processing Utilities
    │   │   ├── SignalUtilities.java  # 信号工具类 / Signal Utilities
    │   │   ├── analysis/             # 信号分析 / Signal Analysis
    │   │   │   ├── ISignalAnalyzer.java    # 信号分析接口 / Signal Analysis Interface
    │   │   │   └── SpectrumAnalyzer.java   # 频谱分析器 / Spectrum Analyzer
    │   │   ├── core/                 # 核心接口 / Core Interfaces
    │   │   │   ├── AbstractSignalProcessor.java # 抽象信号处理器 / Abstract Signal Processor
    │   │   │   ├── Complex.java           # 复数类 / Complex Number Class
    │   │   │   ├── ISignalProcessor.java   # 信号处理接口 / Signal Processing Interface
    │   │   │   ├── RereDCT.java          # 离散余弦变换 / Discrete Cosine Transform
    │   │   │   ├── RereFFT.java          # 快速傅里叶变换 / Fast Fourier Transform
    │   │   │   ├── RereHilbert.java      # 希尔伯特变换 / Hilbert Transform
    │   │   │   └── SignalProcessingException.java # 信号处理异常 / Signal Processing Exception
    │   │   ├── factory/              # 信号处理器工厂 / Signal Processor Factory
    │   │   │   └── SignalProcessorFactory.java # 信号处理器工厂类 / Signal Processor Factory Class
    │   │   ├── filter/               # 信号滤波器 / Signal Filters
    │   │   │   ├── BandpassFilter.java     # 带通滤波器 / Bandpass Filter
    │   │   │   ├── BandStopFilter.java     # 带阻滤波器 / Band Stop Filter
    │   │   │   ├── BesselFilter.java       # 贝塞尔滤波器 / Bessel Filter
    │   │   │   ├── ButterworthFilter.java  # 巴特沃斯滤波器 / Butterworth Filter
    │   │   │   ├── ChebyshevFilter.java    # 切比雪夫滤波器 / Chebyshev Filter
    │   │   │   ├── EllipticFilter.java     # 椭圆滤波器 / Elliptic Filter
    │   │   │   ├── GaussianFilter.java     # 高斯滤波器 / Gaussian Filter
    │   │   │   ├── ISignalFilter.java      # 信号滤波接口 / Signal Filter Interface
    │   │   │   ├── KalmanFilter.java       # 卡尔曼滤波器 / Kalman Filter
    │   │   │   ├── MedianFilter.java       # 中值滤波器 / Median Filter
    │   │   │   ├── MovingAverageFilter.java # 移动平均滤波器 / Moving Average Filter
    │   │   │   └── WienerFilter.java       # 维纳滤波器 / Wiener Filter
    │   │   ├── generation/           # 信号生成 / Signal Generation
    │   │   │   ├── ISignalGenerator.java   # 信号生成接口 / Signal Generator Interface
    │   │   │   └── SignalGenerator.java    # 信号生成器 / Signal Generator
    │   │   ├── transform/            # 信号变换 / Signal Transforms
    │   │   │   ├── ChirpZTransform.java    # Chirp Z变换 / Chirp Z Transform
    │   │   │   ├── HilbertTransform.java   # 希尔伯特变换 / Hilbert Transform
    │   │   │   ├── ISignalTransform.java   # 信号变换接口 / Signal Transform Interface
    │   │   │   ├── WalshHadamardTransform.java # 沃尔什-哈达玛变换 / Walsh-Hadamard Transform
    │   │   │   ├── WaveletTransform.java   # 小波变换 / Wavelet Transform
    │   │   │   └── ZTransform.java         # Z变换 / Z Transform
    │   │   └── wavele/               # 小波分析 / Wavelet Analysis
    │   │       ├── WaveletAnalysis.java  # 小波分析 / Wavelet Analysis
    │   │       ├── WaveletCoefficients.java # 小波系数 / Wavelet Coefficients
    │   │       ├── WaveletFilters.java   # 小波滤波器 / Wavelet Filters
    │   │       ├── WaveletPlots.java     # 小波绘图 / Wavelet Plots
    │   │       └── WaveletUtilities.java # 小波工具类 / Wavelet Utilities
    │   ├── stats/                # 统计学模块 / Statistics Module
    │   │   ├── Stats.java         # 统计分布工厂类 / Statistical Distribution Factory Class
    │   │   ├── anova/            # 方差分析模块 / ANOVA Module
    │   │   │   ├── ANOVA.java                   # 方差分析 / ANOVA
    │   │   │   ├── ANOVAResult.java             # 方差分析结果 / ANOVA Result
    │   │   │   ├── ANOVATest.java               # 方差分析测试 / ANOVA Test
    │   │   │   ├── RepeatedMeasuresANOVAResult.java # 重复测量方差分析结果 / Repeated Measures ANOVA Result
    │   │   │   └── TwoWayANOVAResult.java       # 双因素方差分析结果 / Two-Way ANOVA Result
    │   │   ├── bayes/             # 贝叶斯统计 / Bayesian Statistics
    │   │   │   ├── mcmc/              # MCMC方法 / MCMC Methods
    │   │   │   │   ├── GibbsSampler.java        # 吉布斯采样器 / Gibbs Sampler
    │   │   │   │   ├── HamiltonianMonteCarlo.java # 哈密尔顿蒙特卡洛 / Hamiltonian Monte Carlo
    │   │   │   │   ├── IMCMCSampler.java        # MCMC采样器接口 / MCMC Sampler Interface
    │   │   │   │   ├── MetropolisHastingsSampler.java # Metropolis-Hastings采样器 / Metropolis-Hastings Sampler
    │   │   │   │   └── NoUTurnSampler.java      # NUTS采样器 / NUTS Sampler
    │   │   │   ├── models/            # 贝叶斯模型 / Bayesian Models
    │   │   │   │   ├── BayesianHierarchicalModel.java # 贝叶斯层次模型 / Bayesian Hierarchical Model
    │   │   │   │   ├── BayesianLogisticRegression.java # 贝叶斯逻辑回归 / Bayesian Logistic Regression
    │   │   │   │   ├── GaussianProcess.java     # 高斯过程 / Gaussian Process
    │   │   │   └── variational/       # 变分推理 / Variational Inference
    │   │   │       ├── AutomaticDifferentiationVI.java # 自动微分变分推理 / Automatic Differentiation VI
    │   │   │       ├── IVariationalInference.java # 变分推理接口 / Variational Inference Interface
    │   │   │       └── MeanFieldVariationalInference.java # 平均场变分推理 / Mean Field Variational Inference
    │   │   ├── distribution/     # 概率分布实现 / Probability Distribution Implementations
    │   │   │   ├── IContinuousDistribution.java # 连续分布接口 / Continuous Distribution Interface
    │   │   │   ├── IDiscreteDistribution.java   # 离散分布接口 / Discrete Distribution Interface
    │   │   │   ├── BetaDistribution.java        # Beta分布 / Beta Distribution
    │   │   │   ├── BernoulliDistribution.java   # 伯努利分布 / Bernoulli Distribution
    │   │   │   ├── BinomialDistribution.java    # 二项分布 / Binomial Distribution
    │   │   │   ├── Chi2Distribution.java        # 卡方分布 / Chi-squared Distribution
    │   │   │   ├── DiscreteUniformDistribution.java # 离散均匀分布 / Discrete Uniform Distribution
    │   │   │   ├── ExponentialDistribution.java # 指数分布 / Exponential Distribution
    │   │   │   ├── FDistribution.java           # F分布 / F-Distribution
    │   │   │   ├── GammaDistribution.java       # Gamma分布 / Gamma Distribution
    │   │   │   ├── GeometricDistribution.java   # 几何分布 / Geometric Distribution
    │   │   │   ├── multiv/              # 多变量分布 / Multivariate Distributions
    │   │   │   │   ├── IMultivariateDistribution.java # 多变量分布接口 / Multivariate Distribution Interface
    │   │   │   │   ├── MultivariateBetaDistribution.java # 多变量Beta分布 / Multivariate Beta Distribution
    │   │   │   │   ├── MultivariateDistributions.java # 多变量分布工厂类 / Multivariate Distributions Factory
    │   │   │   │   ├── MultivariateExponentialDistribution.java # 多变量指数分布 / Multivariate Exponential Distribution
    │   │   │   │   ├── MultivariateNormalDistribution.java # 多变量正态分布 / Multivariate Normal Distribution
    │   │   │   │   ├── MultivariateTDistribution.java # 多变量t分布 / Multivariate t-Distribution
    │   │   │   │   └── MultivariateUniformDistribution.java # 多变量均匀分布 / Multivariate Uniform Distribution
    │   │   │   ├── NegativeBinomialDistribution.java # 负二项分布 / Negative Binomial Distribution
    │   │   │   ├── NormalDistribution.java      # 正态分布 / Normal Distribution
    │   │   │   ├── PoissonDistribution.java     # 泊松分布 / Poisson Distribution
    │   │   │   ├── StudentDistribution.java     # t分布 / Student's t-Distribution
    │   │   │   └── UniformDistribution.java     # 均匀分布 / Uniform Distribution
    │   │   ├── model/            # 统计模型 / Statistical Models
    │   │   │   ├── EMAlgorithm.java             # EM算法 / EM Algorithm
    │   │   │   └── GaussianMixtureModel.java    # 高斯混合模型 / Gaussian Mixture Model
    │   │   └── testing/          # 假设检验模块 / Hypothesis Testing Module
    │   │       ├── HypothesisTesting.java      # 假设检验 / Hypothesis Testing
    │   │       ├── ParameterEstimation.java    # 参数估计 / Parameter Estimation
    │   │       └── TestingResult.java          # 检验结果 / Testing Result
    │   ├── timeseries/           # 时间序列分析模块 / Time Series Analysis Module
    │   │   ├── AnalysisResult.java     # 分析结果 / Analysis Result
    │   │   ├── CointegrationAnalysis.java # 协整分析 / Cointegration Analysis
    │   │   ├── DecompositionResult.java # 分解结果 / Decomposition Result
    │   │   ├── Series.java             # 时间序列类 / Time Series Class
    │   │   ├── TimeSeriesAnalyzer.java # 时间序列分析器 / Time Series Analyzer
    │   │   ├── TimeSeriesData.java     # 时间序列数据 / Time Series Data
    │   │   ├── TimeSeriesDecomposition.java # 时间序列分解 / Time Series Decomposition
    │   │   ├── TimeSeriesFiltering.java # 时间序列滤波 / Time Series Filtering
    │   │   ├── TimeSeriesForecasting.java # 时间序列预测 / Time Series Forecasting
    │   │   ├── TimeSeriesPlots.java     # 时间序列绘图 / Time Series Plots
    │   │   ├── TimeSeriesUnifiedExample.java # 时间序列统一示例 / Time Series Unified Example
    │   │   ├── TimeSeriesUtils.java     # 时间序列工具类 / Time Series Utilities
    │   │   └── model/               # 时间序列模型 / Time Series Models
    │   │       ├── ARIMADiagnostics.java     # ARIMA诊断 / ARIMA Diagnostics
    │   │       ├── ExponentialSmoothingModels.java # 指数平滑模型 / Exponential Smoothing Models
    │   │       ├── GARCHModel.java           # GARCH模型 / GARCH Model
    │   │       ├── ITimeSeriesDiagnostics.java # 时间序列诊断接口 / Time Series Diagnostics Interface
    │   │       ├── ITimeSeriesForecastResult.java # 时间序列预测结果接口 / Time Series Forecast Result Interface
    │   │       ├── ITimeSeriesModel.java      # 时间序列模型接口 / Time Series Model Interface
    │   │       ├── StateSpaceModel.java      # 状态空间模型 / State Space Model
    │   │       ├── TimeSeriesForecastResult.java # 时间序列预测结果 / Time Series Forecast Result
    │   │       ├── TimeSeriesModelFactory.java # 时间序列模型工厂 / Time Series Model Factory
    │   │       ├── UnifiedARIMAModel.java     # 统一ARIMA模型 / Unified ARIMA Model
    │   │       └── VARModel.java             # VAR模型 / VAR Model
    │   ├── util/                 # 工具类 / Utilities
    │   │   └── RerePrecision.java # 精度工具 / Precision Utility
    │   └── viz/                  # 数据可视化模块 / Data Visualization Module
    │       ├── AxisTicks.java        # 坐标轴刻度类 / Axis Ticks Class
    │       ├── ColorPalette.java     # 颜色调色板类 / Color Palette Class
    │       ├── IPlot.java            # 绘图接口 / Plotting Interface
    │       ├── PlotException.java    # 绘图异常类 / Plotting Exception Class
    │       ├── PlotStyle.java        # 绘图样式类 / Plot Style Class
    │       ├── Plots.java            # 绘图工厂类 / Plotting Factory Class
    │       ├── RerePlot.java         # 绘图实现类 / Plotting Implementation Class
    │       ├── SeabornStyleMapper.java # Seaborn样式映射器 / Seaborn Style Mapper
    │       ├── StyleConverter.java   # 样式转换器 / Style Converter
    │       ├── StyleExpression.java  # 样式表达式 / Style Expression
    │       ├── ThemeManager.java     # 主题管理器 / Theme Manager
    │       └── UniversalStyleApplier.java # 通用样式应用器 / Universal Style Applier
    ├── music/                    # 音乐处理模块 / Music Processing Module
    │   ├── MusicPlots.java       # 音乐绘图 / Music Plots
    │   ├── MusicRadarDemo.java   # 音乐雷达演示 / Music Radar Demo
    │   ├── Musics.java           # 音乐处理入口类 / Music Processing Entry Class
    │   ├── analysis/             # 音乐分析 / Music Analysis
    │   │   ├── AdvancedMusicAnalyzer.java # 高级音乐分析器 / Advanced Music Analyzer
    │   │   ├── BasicMusicAnalyzer.java    # 基础音乐分析器 / Basic Music Analyzer
    │   │   ├── ComprehensiveMusicAnalyzer.java # 综合音乐分析器 / Comprehensive Music Analyzer
    │   │   ├── ConfidenceCalculator.java  # 置信度计算器 / Confidence Calculator
    │   │   ├── IMusicAnalyzer.java        # 音乐分析接口 / Music Analyzer Interface
    │   │   ├── MusicDetectionResult.java  # 音乐检测结果 / Music Detection Result
    │   │   ├── PerformanceMonitor.java    # 性能监控器 / Performance Monitor
    │   │   ├── StandardizedConfidenceCalculator.java # 标准化置信度计算器 / Standardized Confidence Calculator
    │   │   ├── UnifiedMusicAnalysisResult.java # 统一音乐分析结果 / Unified Music Analysis Result
    │   │   ├── advanced/          # 高级分析 / Advanced Analysis
    │   │   │   ├── ComplexityAnalysisResult.java # 复杂度分析结果 / Complexity Analysis Result
    │   │   │   ├── ComplexityAnalyzer.java # 复杂度分析器 / Complexity Analyzer
    │   │   │   ├── EmotionAnalysisResult.java # 情感分析结果 / Emotion Analysis Result
    │   │   │   ├── EmotionAnalyzer.java # 情感分析器 / Emotion Analyzer
    │   │   │   ├── GenreAnalysisResult.java # 类型分析结果 / Genre Analysis Result
    │   │   │   ├── GenreAnalyzer.java # 类型分析器 / Genre Analyzer
    │   │   │   └── IAdvancedAnalyzer.java # 高级分析接口 / Advanced Analyzer Interface
    │   │   ├── basic/             # 基础分析 / Basic Analysis
    │   │   │   ├── BeatAnalyzerImpl.java # 节拍分析器实现 / Beat Analyzer Implementation
    │   │   │   ├── BeatDetectionResult.java # 节拍检测结果 / Beat Detection Result
    │   │   │   ├── ChordAnalyzerImpl.java # 和弦分析器实现 / Chord Analyzer Implementation
    │   │   │   ├── ChordDetectionResult.java # 和弦检测结果 / Chord Detection Result
    │   │   │   ├── IBeatAnalyzer.java # 节拍分析接口 / Beat Analyzer Interface
    │   │   │   ├── IChordAnalyzer.java # 和弦分析接口 / Chord Analyzer Interface
    │   │   │   ├── IKeyAnalyzer.java # 调性分析接口 / Key Analyzer Interface
    │   │   │   ├── KeyAnalyzerImpl.java # 调性分析器实现 / Key Analyzer Implementation
    │   │   │   ├── KeyDetectionResult.java # 调性检测结果 / Key Detection Result
    │   │   ├── feature/           # 特征提取 / Feature Extraction
    │   │   │   ├── ExpressivenessFeatureResult.java # 表现力特征结果 / Expressiveness Feature Result
    │   │   │   ├── FeatureExtractorImpl.java # 特征提取器实现 / Feature Extractor Implementation
    │   │   │   ├── IFeatureExtractor.java # 特征提取接口 / Feature Extractor Interface
    │   │   │   ├── MusicFeatureResult.java # 音乐特征结果 / Music Feature Result
    │   │   │   ├── RhythmFeatureResult.java # 节奏特征结果 / Rhythm Feature Result
    │   │   │   └── StructureFeatureResult.java # 结构特征结果 / Structure Feature Result
    │   │   │   └── TonalFeatureResult.java # 调性特征结果 / Tonal Feature Result
    │   ├── core/                 # 核心类 / Core Classes
    │   │   └── MusicUtil.java    # 音乐工具类 / Music Utilities Class
    │   ├── factory/              # 音乐工厂 / Music Factory
    │   │   └── MusicComponentFactory.java # 音乐组件工厂 / Music Component Factory
    │   ├── filter/               # 音乐滤波器 / Music Filters
    │   │   └── IMusicFilter.java # 音乐滤波接口 / Music Filter Interface
    │   ├── generation/           # 音乐生成 / Music Generation
    │   │   ├── ChordGenerator.java # 和弦生成器 / Chord Generator
    │   │   ├── IMusicGenerator.java # 音乐生成接口 / Music Generator Interface
    │   │   ├── IntervalGenerator.java # 音程生成器 / Interval Generator
    │   │   └── ScaleGenerator.java # 音阶生成器 / Scale Generator
    │   ├── processing/           # 音乐处理 / Music Processing
    │   │   ├── Harmonizer.java       # 和声器 / Harmonizer
    │   │   ├── IMusicProcessor.java  # 音乐处理器接口 / Music Processor Interface
    │   │   ├── MusicTheoryProcessor.java # 音乐理论处理器 / Music Theory Processor
    │   │   ├── Quantizer.java        # 量化器 / Quantizer
    │   │   └── Transposer.java       # 移调器 / Transposer
    │   └── theory/               # 音乐理论 / Music Theory
    │       ├── ChordTheory.java  # 和弦理论 / Chord Theory
    │       ├── IntervalTheory.java # 音程理论 / Interval Theory
    │       ├── KeyTheory.java    # 调性理论 / Key Theory
    │       └── ScaleTheory.java  # 音阶理论 / Scale Theory
    └── util/                     # 工具类模块 / Utility Module
        ├── RereCollectionUtil.java   # 集合工具类 / Collection Utility Class
        ├── RereExecutor.java         # 执行器工具类 / Executor Utility Class
        ├── RereTree.java             # 树结构工具类 / Tree Structure Utility Class
        ├── RereTreeNode.java         # 树节点工具类 / Tree Node Utility Class
        ├── StringUtils.java          # 字符串工具类 / String Utility Class
        ├── Tuple2.java               # 二元组 / Tuple2
        ├── Tuple3.java               # 三元组 / Tuple3
        ├── Tuple4.java               # 四元组 / Tuple4
        ├── Tuple5.java               # 五元组 / Tuple5
        ├── Tuple6.java               # 六元组 / Tuple6
        ├── Tuple7.java               # 七元组 / Tuple7
        ├── Tuple8.java               # 八元组 / Tuple8
        └── Tuple9.java               # 九元组 / Tuple9
```
