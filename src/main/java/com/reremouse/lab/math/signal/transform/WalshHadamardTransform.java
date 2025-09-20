package com.reremouse.lab.math.signal.transform;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.AbstractSignalProcessor;
import com.reremouse.lab.math.signal.core.SignalProcessingException;

/**
 * Walsh-Hadamard变换实现类 / Walsh-Hadamard Transform Implementation Class
 * <p>
 * 实现快速Walsh-Hadamard变换（FWHT），这是一种基于Hadamard矩阵的正交变换。
 * Walsh-Hadamard变换在数字信号处理、图像压缩、码分多址通信等领域有重要应用。
 * </p>
 * <p>
 * Implements Fast Walsh-Hadamard Transform (FWHT), an orthogonal transform based on Hadamard matrices.
 * Walsh-Hadamard transform has important applications in digital signal processing, image compression, CDMA communications, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WalshHadamardTransform extends AbstractSignalProcessor<Double> implements ISignalTransform<Double, IVector<Double>> {
    
    /**
     * Walsh函数排序方式枚举 / Walsh Function Ordering Enum
     */
    public enum WalshOrdering {
        NATURAL("自然排序", "Natural Ordering"),
        SEQUENCY("序列排序", "Sequency Ordering"),
        DYADIC("二进制排序", "Dyadic Ordering");
        
        private final String chineseName;
        private final String englishName;
        
        WalshOrdering(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    private WalshOrdering ordering;
    private boolean normalized;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用默认参数初始化Walsh-Hadamard变换。
     * Initialize Walsh-Hadamard transform with default parameters.
     * </p>
     */
    public WalshHadamardTransform() {
        super("Walsh-Hadamard Transform", "1.0.0");
        this.ordering = WalshOrdering.NATURAL;
        this.normalized = true;
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用指定参数初始化Walsh-Hadamard变换。
     * Initialize Walsh-Hadamard transform with specified parameters.
     * </p>
     *
     * @param ordering Walsh函数排序方式 / Walsh function ordering
     * @param normalized 是否归一化 / Whether to normalize
     */
    public WalshHadamardTransform(WalshOrdering ordering, boolean normalized) {
        super("Walsh-Hadamard Transform", "1.0.0");
        this.ordering = ordering;
        this.normalized = normalized;
    }
    
    /**
     * 计算快速Walsh-Hadamard变换 / Calculate Fast Walsh-Hadamard Transform
     * <p>
     * 使用快速算法计算Walsh-Hadamard变换，时间复杂度为O(N log N)。
     * Calculate Walsh-Hadamard transform using fast algorithm with O(N log N) complexity.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return Walsh-Hadamard变换结果 / Walsh-Hadamard transform result
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    @Override
    public IVector<Double> forward(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        
        int n = signal.length();
        
        // 检查长度是否为2的幂 / Check if length is power of 2
        if ((n & (n - 1)) != 0) {
            throw new SignalProcessingException("信号长度必须是2的幂 / Signal length must be power of 2");
        }
        
        // 复制输入信号 / Copy input signal
        IVector<Double> result = signal.copy();
        
        // 执行快速Walsh-Hadamard变换 / Perform fast Walsh-Hadamard transform
        result = performFWHT(result);
        
        // 根据排序方式重新排列 / Reorder according to ordering scheme
        if (ordering != WalshOrdering.NATURAL) {
            result = reorderResult(result, ordering);
        }
        
        // 归一化 / Normalize
        if (normalized) {
            double normFactor = 1.0 / Math.sqrt(n);
            result = result.multiplyScalar(normFactor);
        }
        
        return result;
    }
    
    /**
     * 计算逆Walsh-Hadamard变换 / Calculate Inverse Walsh-Hadamard Transform
     * <p>
     * Walsh-Hadamard变换是自逆的，即逆变换等于正变换（除了归一化因子）。
     * Walsh-Hadamard transform is self-inverse, i.e., inverse transform equals forward transform (except normalization factor).
     * </p>
     *
     * @param transformed Walsh-Hadamard变换结果 / Walsh-Hadamard transform result
     * @return 原始信号 / Original signal
     * @throws SignalProcessingException 逆变换过程中发生错误时抛出 / Thrown when errors occur during inverse transform
     */
    @Override
    public IVector<Double> inverse(IVector<Double> transformed) throws SignalProcessingException {
        if (transformed == null || transformed.length() == 0) {
            throw new SignalProcessingException("输入变换结果不能为空 / Input transform result cannot be empty");
        }
        
        int n = transformed.length();
        
        // 检查长度是否为2的幂 / Check if length is power of 2
        if ((n & (n - 1)) != 0) {
            throw new SignalProcessingException("变换结果长度必须是2的幂 / Transform result length must be power of 2");
        }
        
        // 复制输入 / Copy input
        IVector<Double> result = transformed.copy();
        
        // 如果使用了排序，先恢复自然排序 / If ordering was used, restore natural ordering first
        if (ordering != WalshOrdering.NATURAL) {
            result = restoreNaturalOrdering(result, ordering);
        }
        
        // 执行快速Walsh-Hadamard变换（自逆性质） / Perform fast Walsh-Hadamard transform (self-inverse property)
        result = performFWHT(result);
        
        // 归一化 / Normalize
        if (normalized) {
            double normFactor = 1.0 / Math.sqrt(n);
            result = result.multiplyScalar(normFactor);
        } else {
            // 如果正变换没有归一化，逆变换需要除以N / If forward transform wasn't normalized, inverse transform needs division by N
            result = result.multiplyScalar(1.0 / n);
        }
        
        return result;
    }
    
    /**
     * 执行快速Walsh-Hadamard变换核心算法 / Perform core Fast Walsh-Hadamard Transform algorithm
     * <p>
     * 使用分治算法实现快速Walsh-Hadamard变换。
     * Implement fast Walsh-Hadamard transform using divide-and-conquer algorithm.
     * </p>
     */
    private IVector<Double> performFWHT(IVector<Double> data) {
        int n = data.length();
        IVector<Double> result = data.copy();
        
        // 使用蝶形运算实现快速变换 / Use butterfly operations for fast transform
        for (int step = 1; step < n; step <<= 1) {
            for (int i = 0; i < n; i += step << 1) {
                for (int j = 0; j < step; j++) {
                    double u = result.get(i + j);
                    double v = result.get(i + j + step);
                    result.set(i + j, u + v);
                    result.set(i + j + step, u - v);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 根据排序方式重新排列结果 / Reorder result according to ordering scheme
     */
    private IVector<Double> reorderResult(IVector<Double> data, WalshOrdering ordering) {
        int n = data.length();
        IVector<Double> reordered = Linalg.zeros(n);
        
        switch (ordering) {
            case SEQUENCY:
                // 按序列数排序 / Order by sequency number
                for (int i = 0; i < n; i++) {
                    int sequencyIndex = getSequencyIndex(i, n);
                    reordered.set(sequencyIndex, data.get(i));
                }
                break;
                
            case DYADIC:
                // 按二进制反序排序 / Order by bit-reversal
                for (int i = 0; i < n; i++) {
                    int bitReversedIndex = bitReverse(i, Integer.numberOfTrailingZeros(n));
                    reordered.set(bitReversedIndex, data.get(i));
                }
                break;
                
            default:
                reordered = data.copy();
                break;
        }
        
        return reordered;
    }
    
    /**
     * 恢复自然排序 / Restore natural ordering
     */
    private IVector<Double> restoreNaturalOrdering(IVector<Double> data, WalshOrdering ordering) {
        int n = data.length();
        IVector<Double> restored = Linalg.zeros(n);
        
        switch (ordering) {
            case SEQUENCY:
                // 从序列数排序恢复 / Restore from sequency ordering
                for (int i = 0; i < n; i++) {
                    int sequencyIndex = getSequencyIndex(i, n);
                    restored.set(i, data.get(sequencyIndex));
                }
                break;
                
            case DYADIC:
                // 从二进制反序排序恢复 / Restore from bit-reversal ordering
                for (int i = 0; i < n; i++) {
                    int bitReversedIndex = bitReverse(i, Integer.numberOfTrailingZeros(n));
                    restored.set(i, data.get(bitReversedIndex));
                }
                break;
                
            default:
                restored = data.copy();
                break;
        }
        
        return restored;
    }
    
    /**
     * 计算序列索引 / Calculate sequency index
     * <p>
     * 序列数是Walsh函数在一个周期内符号变化的次数。
     * Sequency is the number of sign changes of Walsh function within one period.
     * </p>
     */
    private int getSequencyIndex(int index, int n) {
        // 这里使用简化的映射，实际实现可能需要更复杂的序列数计算
        // Simplified mapping used here, actual implementation may need more complex sequency calculation
        int logN = Integer.numberOfTrailingZeros(n);
        int sequency = 0;
        
        // 计算格雷码的汉明重量 / Calculate Hamming weight of Gray code
        int gray = index ^ (index >> 1);
        for (int i = 0; i < logN; i++) {
            if ((gray & (1 << i)) != 0) {
                sequency++;
            }
        }
        
        return sequency;
    }
    
    /**
     * 计算二进制反序 / Calculate bit reversal
     */
    private int bitReverse(int num, int bits) {
        int result = 0;
        for (int i = 0; i < bits; i++) {
            result = (result << 1) | (num & 1);
            num >>= 1;
        }
        return result;
    }
    
    /**
     * 计算Walsh函数值 / Calculate Walsh function value
     * <p>
     * 计算指定索引的Walsh函数在给定点的值。
     * Calculate value of Walsh function with specified index at given point.
     * </p>
     *
     * @param n Walsh函数索引 / Walsh function index
     * @param t 时间点 / Time point
     * @param N 总点数 / Total number of points
     * @return Walsh函数值 / Walsh function value
     */
    public static double walshFunction(int n, int t, int N) {
        int product = 1;
        int mask = 1;
        
        while (mask < N) {
            if ((n & mask) != 0) {
                if (((2 * t * mask) / N) % 2 == 1) {
                    product *= -1;
                }
            }
            mask <<= 1;
        }
        
        return product;
    }
    
    /**
     * 生成Hadamard矩阵 / Generate Hadamard matrix
     * <p>
     * 生成n×n的Hadamard矩阵，其中n必须是2的幂。
     * Generate n×n Hadamard matrix where n must be power of 2.
     * </p>
     *
     * @param n 矩阵大小 / Matrix size
     * @return Hadamard矩阵 / Hadamard matrix
     * @throws SignalProcessingException 矩阵大小无效时抛出 / Thrown when matrix size is invalid
     */
    public static double[][] generateHadamardMatrix(int n) throws SignalProcessingException {
        if ((n & (n - 1)) != 0) {
            throw new SignalProcessingException("Hadamard矩阵大小必须是2的幂 / Hadamard matrix size must be power of 2");
        }
        
        double[][] hadamard = new double[n][n];
        
        // 使用递归构造Hadamard矩阵 / Construct Hadamard matrix recursively
        if (n == 1) {
            hadamard[0][0] = 1;
        } else {
            double[][] half = generateHadamardMatrix(n / 2);
            
            // 构造2×2块矩阵 / Construct 2×2 block matrix
            // H_n = [H_{n/2}  H_{n/2}]
            //       [H_{n/2} -H_{n/2}]
            for (int i = 0; i < n / 2; i++) {
                for (int j = 0; j < n / 2; j++) {
                    hadamard[i][j] = half[i][j];
                    hadamard[i][j + n / 2] = half[i][j];
                    hadamard[i + n / 2][j] = half[i][j];
                    hadamard[i + n / 2][j + n / 2] = -half[i][j];
                }
            }
        }
        
        return hadamard;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return forward(input);
    }
    
    @Override
    public WalshHadamardTransform clone() {
        return new WalshHadamardTransform(ordering, normalized);
    }
    
    // Getters and setters
    public WalshOrdering getOrdering() { return ordering; }
    public void setOrdering(WalshOrdering ordering) { this.ordering = ordering; }
    
    public boolean isNormalized() { return normalized; }
    public void setNormalized(boolean normalized) { this.normalized = normalized; }
}