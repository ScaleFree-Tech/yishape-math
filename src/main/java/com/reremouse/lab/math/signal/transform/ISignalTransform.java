package com.reremouse.lab.math.signal.transform;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.Complex;
import com.reremouse.lab.math.signal.core.ISignalProcessor;
import com.reremouse.lab.math.signal.core.SignalProcessingException;

/**
 * 信号变换接口 / Signal Transform Interface
 * <p>
 * 定义所有信号变换操作的基础接口，包括正变换和逆变换。
 * 使用策略模式支持不同的变换算法实现。
 * </p>
 * <p>
 * Defines the base interface for all signal transform operations including forward and inverse transforms.
 * Uses Strategy pattern to support different transform algorithm implementations.
 * </p>
 *
 * @param <T> 输入信号数据类型 / Input signal data type
 * @param <R> 输出变换结果类型 / Output transform result type
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ISignalTransform<T extends Number, R> extends ISignalProcessor<T> {
    
    /**
     * 正向变换 / Forward transform
     * <p>
     * 将信号从时域转换到变换域。
     * Transform signal from time domain to transform domain.
     * </p>
     *
     * @param signal 输入时域信号 / Input time domain signal
     * @return 变换域结果 / Transform domain result
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    R forward(IVector<T> signal) throws SignalProcessingException;
    
    /**
     * 逆变换 / Inverse transform
     * <p>
     * 将信号从变换域转换回时域。
     * Transform signal from transform domain back to time domain.
     * </p>
     *
     * @param transformed 变换域信号 / Transform domain signal
     * @return 时域结果 / Time domain result
     * @throws SignalProcessingException 逆变换过程中发生错误时抛出 / Thrown when errors occur during inverse transform
     */
    IVector<T> inverse(R transformed) throws SignalProcessingException;
    
    /**
     * 二维正向变换 / 2D Forward transform
     * <p>
     * 将二维信号从空间域转换到变换域。
     * Transform 2D signal from spatial domain to transform domain.
     * </p>
     *
     * @param signal 输入二维信号 / Input 2D signal
     * @return 二维变换域结果 / 2D transform domain result
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    default Object forward2D(IMatrix<T> signal) throws SignalProcessingException {
        throw new UnsupportedOperationException("2D forward transform not supported");
    }
    
    /**
     * 二维逆变换 / 2D Inverse transform
     * <p>
     * 将二维信号从变换域转换回空间域。
     * Transform 2D signal from transform domain back to spatial domain.
     * </p>
     *
     * @param transformed 二维变换域信号 / 2D transform domain signal
     * @return 二维空间域结果 / 2D spatial domain result
     * @throws SignalProcessingException 逆变换过程中发生错误时抛出 / Thrown when errors occur during inverse transform
     */
    default IMatrix<T> inverse2D(Object transformed) throws SignalProcessingException {
        throw new UnsupportedOperationException("2D inverse transform not supported");
    }
    
    /**
     * 获取变换核大小 / Get transform kernel size
     * <p>
     * 返回变换操作所需的最小信号长度。
     * Return minimum signal length required for transform operation.
     * </p>
     *
     * @return 最小信号长度 / Minimum signal length
     */
    default int getMinimumSignalLength() {
        return 1;
    }
    
    /**
     * 检查信号长度是否支持 / Check if signal length is supported
     * <p>
     * 检查给定的信号长度是否适合当前变换算法。
     * Check if given signal length is suitable for current transform algorithm.
     * </p>
     *
     * @param length 信号长度 / Signal length
     * @return 是否支持 / Whether supported
     */
    default boolean isLengthSupported(int length) {
        return length >= getMinimumSignalLength();
    }
    
    /**
     * 获取推荐的信号长度 / Get recommended signal length
     * <p>
     * 为了最佳性能，返回推荐的信号长度（例如2的幂）。
     * Return recommended signal length for optimal performance (e.g., power of 2).
     * </p>
     *
     * @param originalLength 原始信号长度 / Original signal length
     * @return 推荐长度 / Recommended length
     */
    default int getRecommendedLength(int originalLength) {
        return originalLength;
    }
    
    /**
     * 默认实现process方法 / Default implementation of process method
     * <p>
     * 使用正向变换作为默认的process实现。
     * Use forward transform as default process implementation.
     * </p>
     */
    @Override
    @SuppressWarnings("unchecked")
    default IVector<T> process(IVector<T> input) throws SignalProcessingException {
        R result = forward(input);
        if (result instanceof IVector) {
            return (IVector<T>) result;
        }
        throw new SignalProcessingException("Transform result cannot be cast to IVector<T>");
    }
}