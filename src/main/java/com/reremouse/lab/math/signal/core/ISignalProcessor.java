package com.reremouse.lab.math.signal.core;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;

/**
 * 信号处理器基础接口 / Signal Processor Base Interface
 * <p>
 * 定义所有信号处理操作的基础接口，使用泛型支持不同的数据类型。
 * 遵循策略模式，允许在运行时选择不同的算法实现。
 * </p>
 * <p>
 * Defines the base interface for all signal processing operations with generic support for different data types.
 * Follows Strategy pattern to allow selection of different algorithm implementations at runtime.
 * </p>
 *
 * @param <T> 信号数据类型 / Signal data type
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ISignalProcessor<T extends Number> {
    
    /**
     * 处理一维信号 / Process 1D signal
     * <p>
     * 对输入的一维信号进行处理，返回处理后的结果。
     * Process input 1D signal and return the processed result.
     * </p>
     *
     * @param input 输入信号向量 / Input signal vector
     * @return 处理后的信号向量 / Processed signal vector
     * @throws SignalProcessingException 当处理过程中发生错误时抛出 / Thrown when errors occur during processing
     */
    IVector<T> process(IVector<T> input) throws SignalProcessingException;
    
    /**
     * 处理二维信号 / Process 2D signal
     * <p>
     * 对输入的二维信号进行处理，返回处理后的结果。
     * Process input 2D signal and return the processed result.
     * </p>
     *
     * @param input 输入信号矩阵 / Input signal matrix
     * @return 处理后的信号矩阵 / Processed signal matrix
     * @throws SignalProcessingException 当处理过程中发生错误时抛出 / Thrown when errors occur during processing
     */
    default IMatrix<T> process(IMatrix<T> input) throws SignalProcessingException {
        throw new UnsupportedOperationException("2D signal processing not supported by this processor");
    }
    
    /**
     * 验证输入参数 / Validate input parameters
     * <p>
     * 验证输入信号和处理器配置是否有效。
     * Validate if input signal and processor configuration are valid.
     * </p>
     *
     * @param input 输入信号向量 / Input signal vector
     * @return 验证是否通过 / Whether validation passes
     */
    boolean validateInput(IVector<T> input);
    
    /**
     * 获取处理器名称 / Get processor name
     * <p>
     * 返回处理器的名称，用于识别和调试。
     * Return processor name for identification and debugging.
     * </p>
     *
     * @return 处理器名称 / Processor name
     */
    String getName();
    
    /**
     * 获取处理器版本 / Get processor version
     * <p>
     * 返回处理器的版本信息。
     * Return processor version information.
     * </p>
     *
     * @return 版本信息 / Version information
     */
    default String getVersion() {
        return "1.0.0";
    }
    
    /**
     * 克隆处理器 / Clone processor
     * <p>
     * 创建当前处理器的深拷贝。
     * Create a deep copy of current processor.
     * </p>
     *
     * @return 克隆的处理器 / Cloned processor
     */
    ISignalProcessor<T> clone();
}