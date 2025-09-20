package com.reremouse.lab.math.signal.core;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;

/**
 * 抽象信号处理器基类 / Abstract Signal Processor Base Class
 * <p>
 * 提供信号处理器的通用功能实现，包括参数验证、错误处理等。
 * 使用模板方法模式定义通用的处理流程。
 * </p>
 * <p>
 * Provides common functionality implementation for signal processors including parameter validation and error handling.
 * Uses Template Method pattern to define common processing workflow.
 * </p>
 *
 * @param <T> 信号数据类型 / Signal data type
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public abstract class AbstractSignalProcessor<T extends Number> implements ISignalProcessor<T> {
    
    protected String name;
    protected String version;
    protected boolean initialized = false;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 初始化信号处理器的基本信息。
     * Initialize basic information of signal processor.
     * </p>
     *
     * @param name 处理器名称 / Processor name
     * @param version 处理器版本 / Processor version
     */
    protected AbstractSignalProcessor(String name, String version) {
        this.name = name;
        this.version = version;
        this.initialized = false;
    }
    
    /**
     * 模板方法：处理信号 / Template method: process signal
     * <p>
     * 定义标准的信号处理流程：预处理 -> 处理 -> 后处理。
     * Define standard signal processing workflow: preprocess -> process -> postprocess.
     * </p>
     */
    @Override
    public final IVector<T> process(IVector<T> input) throws SignalProcessingException {
        // 1. 验证输入 / Validate input
        if (!validateInput(input)) {
            throw new SignalProcessingException("输入信号验证失败 / Input signal validation failed");
        }
        
        // 2. 预处理 / Preprocess
        IVector<T> preprocessed = preprocess(input);
        
        // 3. 核心处理 / Core processing
        IVector<T> processed = doProcess(preprocessed);
        
        // 4. 后处理 / Postprocess
        IVector<T> result = postprocess(processed);
        
        // 5. 验证输出 / Validate output
        if (!validateOutput(result)) {
            throw new SignalProcessingException("输出信号验证失败 / Output signal validation failed");
        }
        
        return result;
    }
    
    /**
     * 模板方法：处理二维信号 / Template method: process 2D signal
     */
    @Override
    public IMatrix<T> process(IMatrix<T> input) throws SignalProcessingException {
        // 1. 验证输入 / Validate input
        if (!validateInput2D(input)) {
            throw new SignalProcessingException("输入2D信号验证失败 / Input 2D signal validation failed");
        }
        
        // 2. 预处理 / Preprocess
        IMatrix<T> preprocessed = preprocess2D(input);
        
        // 3. 核心处理 / Core processing
        IMatrix<T> processed = doProcess2D(preprocessed);
        
        // 4. 后处理 / Postprocess
        IMatrix<T> result = postprocess2D(processed);
        
        // 5. 验证输出 / Validate output
        if (!validateOutput2D(result)) {
            throw new SignalProcessingException("输出2D信号验证失败 / Output 2D signal validation failed");
        }
        
        return result;
    }
    
    /**
     * 预处理步骤 / Preprocessing step
     * <p>
     * 子类可以重写此方法来实现特定的预处理逻辑。
     * Subclasses can override this method to implement specific preprocessing logic.
     * </p>
     *
     * @param input 输入信号 / Input signal
     * @return 预处理后的信号 / Preprocessed signal
     * @throws SignalProcessingException 预处理过程中发生错误时抛出 / Thrown when errors occur during preprocessing
     */
    protected IVector<T> preprocess(IVector<T> input) throws SignalProcessingException {
        return input; // 默认不进行预处理 / Default: no preprocessing
    }
    
    /**
     * 核心处理步骤 / Core processing step
     * <p>
     * 子类必须实现此方法来定义具体的处理逻辑。
     * Subclasses must implement this method to define specific processing logic.
     * </p>
     *
     * @param input 预处理后的信号 / Preprocessed signal
     * @return 处理后的信号 / Processed signal
     * @throws SignalProcessingException 处理过程中发生错误时抛出 / Thrown when errors occur during processing
     */
    protected abstract IVector<T> doProcess(IVector<T> input) throws SignalProcessingException;
    
    /**
     * 后处理步骤 / Postprocessing step
     * <p>
     * 子类可以重写此方法来实现特定的后处理逻辑。
     * Subclasses can override this method to implement specific postprocessing logic.
     * </p>
     *
     * @param output 处理后的信号 / Processed signal
     * @return 后处理后的信号 / Postprocessed signal
     * @throws SignalProcessingException 后处理过程中发生错误时抛出 / Thrown when errors occur during postprocessing
     */
    protected IVector<T> postprocess(IVector<T> output) throws SignalProcessingException {
        return output; // 默认不进行后处理 / Default: no postprocessing
    }
    
    // 2D信号处理方法 / 2D signal processing methods
    protected IMatrix<T> preprocess2D(IMatrix<T> input) throws SignalProcessingException {
        return input;
    }
    
    protected IMatrix<T> doProcess2D(IMatrix<T> input) throws SignalProcessingException {
        throw new UnsupportedOperationException("2D signal processing not implemented");
    }
    
    protected IMatrix<T> postprocess2D(IMatrix<T> output) throws SignalProcessingException {
        return output;
    }
    
    /**
     * 验证输入信号 / Validate input signal
     */
    @Override
    public boolean validateInput(IVector<T> input) {
        if (input == null) {
            return false;
        }
        if (input.length() <= 0) {
            return false;
        }
        return validateSpecificInput(input);
    }
    
    /**
     * 验证2D输入信号 / Validate 2D input signal
     */
    protected boolean validateInput2D(IMatrix<T> input) {
        if (input == null) {
            return false;
        }
        if (input.rows() <= 0 || input.cols() <= 0) {
            return false;
        }
        return validateSpecificInput2D(input);
    }
    
    /**
     * 验证输出信号 / Validate output signal
     */
    protected boolean validateOutput(IVector<T> output) {
        if (output == null) {
            return false;
        }
        if (output.length() <= 0) {
            return false;
        }
        return validateSpecificOutput(output);
    }
    
    /**
     * 验证2D输出信号 / Validate 2D output signal
     */
    protected boolean validateOutput2D(IMatrix<T> output) {
        if (output == null) {
            return false;
        }
        if (output.rows() <= 0 || output.cols() <= 0) {
            return false;
        }
        return validateSpecificOutput2D(output);
    }
    
    /**
     * 子类特定的输入验证 / Subclass-specific input validation
     * <p>
     * 子类可以重写此方法来实现特定的输入验证逻辑。
     * Subclasses can override this method to implement specific input validation logic.
     * </p>
     *
     * @param input 输入信号 / Input signal
     * @return 验证是否通过 / Whether validation passes
     */
    protected boolean validateSpecificInput(IVector<T> input) {
        return true; // 默认通过验证 / Default: pass validation
    }
    
    protected boolean validateSpecificInput2D(IMatrix<T> input) {
        return true;
    }
    
    /**
     * 子类特定的输出验证 / Subclass-specific output validation
     */
    protected boolean validateSpecificOutput(IVector<T> output) {
        return true; // 默认通过验证 / Default: pass validation
    }
    
    protected boolean validateSpecificOutput2D(IMatrix<T> output) {
        return true;
    }
    
    /**
     * 初始化处理器 / Initialize processor
     * <p>
     * 子类可以重写此方法来实现特定的初始化逻辑。
     * Subclasses can override this method to implement specific initialization logic.
     * </p>
     *
     * @throws SignalProcessingException 初始化过程中发生错误时抛出 / Thrown when errors occur during initialization
     */
    protected void initialize() throws SignalProcessingException {
        this.initialized = true;
    }
    
    /**
     * 检查是否已初始化 / Check if initialized
     */
    protected void checkInitialized() throws SignalProcessingException {
        if (!initialized) {
            initialize();
        }
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public String getVersion() {
        return version;
    }
    
    /**
     * 获取处理器描述 / Get processor description
     */
    public String getDescription() {
        return String.format("%s v%s - 信号处理器 / Signal Processor", name, version);
    }
    
    /**
     * 检查两个向量长度是否相等 / Check if two vectors have equal length
     */
    protected void checkEqualLength(IVector<T> vector1, IVector<T> vector2) throws SignalProcessingException {
        if (vector1.length() != vector2.length()) {
            throw new SignalProcessingException(
                String.format("向量长度不匹配：%d != %d / Vector length mismatch: %d != %d", 
                    vector1.length(), vector2.length(), vector1.length(), vector2.length()));
        }
    }
    
    /**
     * 检查向量长度是否为2的幂 / Check if vector length is power of 2
     */
    protected void checkPowerOfTwo(IVector<T> vector) throws SignalProcessingException {
        int length = vector.length();
        if ((length & (length - 1)) != 0) {
            throw new SignalProcessingException(
                String.format("向量长度必须是2的幂：%d / Vector length must be power of 2: %d", length, length));
        }
    }
    
    /**
     * 检查频率范围 / Check frequency range
     */
    protected void checkFrequencyRange(double frequency, double samplingRate) throws SignalProcessingException {
        if (frequency < 0 || frequency > samplingRate / 2) {
            throw new SignalProcessingException(
                String.format("频率超出范围 [0, %f]：%f / Frequency out of range [0, %f]: %f", 
                    samplingRate / 2, frequency, samplingRate / 2, frequency));
        }
    }
    
    @Override
    public String toString() {
        return getDescription();
    }
    
    /**
     * 实现clone方法 / Implement clone method
     * <p>
     * 子类应该重写此方法以提供具体的克隆实现。
     * Subclasses should override this method to provide specific clone implementation.
     * </p>
     */
    @Override
    public abstract ISignalProcessor<T> clone();
}