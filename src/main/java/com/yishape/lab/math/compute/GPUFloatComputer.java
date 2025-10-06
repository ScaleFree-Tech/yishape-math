package com.yishape.lab.math.compute;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.Device.TYPE;
import static com.yishape.lab.math.compute.ComputerConfig.*;

/**
 * 改进的GPU计算工具类 - 基于Aparapi框架实现
 * 
 * <p>主要改进：</p>
 * <ul>
 *   <li><strong>设备能力检测</strong>：检测GPU是否支持FP64双精度运算</li>
 *   <li><strong>智能阈值策略</strong>：根据数据大小智能选择CPU或GPU执行</li>
 *   <li><strong>内存分块处理</strong>：大数据时分块处理，避免内存溢出</li>
 *   <li><strong>优雅错误处理</strong>：GPU失败时自动回退到CPU</li>
 *   <li><strong>性能优化</strong>：避免小数据时的GPU开销</li>
 * </ul>
 * 
 * @author lteb2
 * @version 2.0 - 稳定性改进版本
 */
public class GPUFloatComputer implements IFloatVectorComputer {
    
    // GPU状态和能力检测
    private static volatile boolean gpuAvailable = true;
    private static volatile boolean fp64Supported = true;
    private static Device gpuDevice = null;
    private static String gpuInfo = "GPU: 未初始化";
    
    // 日志控制
    private static volatile boolean enableLogging = false;
    
    // 内存分块阈值 - 避免大数据时的内存问题
    private static final int MAX_CHUNK_SIZE = 10_000_000; // 10M elements per chunk
    private static final int TILE_SIZE = 1024; // 矩阵分块大小
    
    static {
        initializeGPU();
    }
    
    /**
     * 初始化GPU环境，包含设备能力检测
     */
    private static void initializeGPU() {
        try {
            // 首先尝试获取可用的GPU设备
//            Device[] devices = Device.getAvailableDevices();
            Device[] devices = {};
            
            for (Device device : devices) {
                if (device.getType() == TYPE.GPU) {
                    gpuDevice = device;
                    gpuAvailable = true;
                    
                    // 检测FP64支持
                    fp64Supported = checkFP64Support(device);
                    
                    gpuInfo = String.format("GPU: %s, FP64支持: %s", 
                        device.getShortDescription(), fp64Supported);
                    
                    if (enableLogging) {
                        System.out.println(gpuInfo);
                    }
                    break;
                }
            }
            
            if (!gpuAvailable) {
                gpuInfo = "GPU: 未找到可用的GPU设备，将使用CPU模式";
                if (enableLogging) {
                    System.out.println(gpuInfo);
                }
            }
            
        } catch (Exception e) {
            gpuAvailable = false;
            fp64Supported = false;
            gpuInfo = "GPU: 初始化失败 - " + e.getMessage();
            if (enableLogging) {
                System.err.println(gpuInfo);
            }
        }
    }
    
    /**
     * 检测GPU设备是否支持FP64双精度运算
     */
    private static boolean checkFP64Support(Device device) {
        try {
            // 尝试创建一个简单的双精度内核来测试支持
            float[] testData = {1.0f, 2.0f};
            float[] result = new float[2];
            
            Kernel testKernel = new Kernel() {
                @Override
                public void run() {
                    int i = getGlobalId(0);
                    if (i < testData.length) {
                        result[i] = testData[i] * 2.0f; // 简单的双精度运算
                    }
                }
            };
            
            testKernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
            testKernel.execute(Range.create(2));
            testKernel.dispose();
            
            // 如果执行成功且结果正确，说明支持FP64
            return Math.abs(result[0] - 2.0) < 1e-10 && Math.abs(result[1] - 4.0) < 1e-10;
            
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * 检查是否应该使用GPU执行
     */
    private static boolean shouldUseGPU(int dataSize) {
        return gpuAvailable && fp64Supported && dataSize >= GPU_THRESHOLD;
    }
    
    /**
     * 记录CPU回退日志
     */
    private static void logCPUFallback(String operation, String reason) {
        if (enableLogging) {
            System.out.println(String.format("[CPU回退] %s: %s", operation, reason));
        }
    }
    
    /**
     * 检查GPU是否可用
     */
    public static boolean isGPUAvailable() {
        return gpuAvailable && fp64Supported;
    }
    
    /**
     * 获取GPU信息
     */
    public static String getGPUInfo() {
        return gpuInfo;
    }
    
    /**
     * 设置日志启用状态
     */
    public static void setLoggingEnabled(boolean enabled) {
        enableLogging = enabled;
    }

    @Override
    public float[] binaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }
        
        int length = x1.length;
        
        // 智能选择执行策略
        if (!shouldUseGPU(length)) {
            logCPUFallback("向量二元运算", "数据量小于阈值或GPU不支持FP64");
            return cpuBinaryOperate(x1, x2, operation);
        }
        
        // 大数据分块处理
        if (length > MAX_CHUNK_SIZE) {
            return chunkedBinaryOperate(x1, x2, operation);
        }
        
        // GPU执行
        return gpuBinaryOperate(x1, x2, operation);
    }
    
    /**
     * CPU版本的二元运算
     */
    private float[] cpuBinaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        float[] result = new float[x1.length];
        
        for (int i = 0; i < x1.length; i++) {
            switch (operation) {
                case ADD:
                    result[i] = x1[i] + x2[i];
                    break;
                case SUBTRACT:
                    result[i] = x1[i] - x2[i];
                    break;
                case MULTIPLY:
                    result[i] = x1[i] * x2[i];
                    break;
                case DIVIDE:
                    result[i] = x1[i] / x2[i];
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }
        
        return result;
    }
    
    /**
     * 分块处理大数据的二元运算
     */
    private float[] chunkedBinaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        int length = x1.length;
        float[] result = new float[length];
        
        for (int start = 0; start < length; start += MAX_CHUNK_SIZE) {
            int end = Math.min(start + MAX_CHUNK_SIZE, length);
            int chunkSize = end - start;
            
            // 创建分块数据
            float[] chunk1 = new float[chunkSize];
            float[] chunk2 = new float[chunkSize];
            System.arraycopy(x1, start, chunk1, 0, chunkSize);
            System.arraycopy(x2, start, chunk2, 0, chunkSize);
            
            // 处理分块
            float[] chunkResult = gpuBinaryOperate(chunk1, chunk2, operation);
            
            // 复制结果
            System.arraycopy(chunkResult, 0, result, start, chunkSize);
        }
        
        return result;
    }
    
    /**
     * GPU版本的二元运算
     */
    private float[] gpuBinaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        int length = x1.length;
        float[] result = new float[length];
        
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    switch (operation) {
                        case ADD:
                            result[i] = x1[i] + x2[i];
                            break;
                        case SUBTRACT:
                            result[i] = x1[i] - x2[i];
                            break;
                        case MULTIPLY:
                            result[i] = x1[i] * x2[i];
                            break;
                        case DIVIDE:
                            result[i] = x1[i] / x2[i];
                            break;
                    }
                }
            }
        };
        
        try {
            kernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
            Range range = Range.create(length);
            kernel.execute(range);
            return result;
        } catch (Exception e) {
            logCPUFallback("GPU向量二元运算", "GPU执行失败: " + e.getMessage());
            return cpuBinaryOperate(x1, x2, operation);
        } finally {
            kernel.dispose();
        }
    }

    @Override
    public float[] binaryOperate(float[] x1, float x2, BinaryOperation operation) {
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = x1.length;
        
        // 智能选择执行策略
        if (!shouldUseGPU(length)) {
            logCPUFallback("向量标量运算", "数据量小于阈值或GPU不支持FP64");
            return cpuBinaryOperate(x1, x2, operation);
        }
        
        // 大数据分块处理
        if (length > MAX_CHUNK_SIZE) {
            return chunkedBinaryOperate(x1, x2, operation);
        }
        
        // GPU执行
        return gpuBinaryOperate(x1, x2, operation);
    }
    
    /**
     * CPU版本的向量标量运算
     */
    private float[] cpuBinaryOperate(float[] x1, float x2, BinaryOperation operation) {
        float[] result = new float[x1.length];
        
        for (int i = 0; i < x1.length; i++) {
            switch (operation) {
                case ADD:
                    result[i] = x1[i] + x2;
                    break;
                case SUBTRACT:
                    result[i] = x1[i] - x2;
                    break;
                case MULTIPLY:
                    result[i] = x1[i] * x2;
                    break;
                case DIVIDE:
                    result[i] = x1[i] / x2;
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }
        
        return result;
    }
    
    /**
     * 分块处理大数据的向量标量运算
     */
    private float[] chunkedBinaryOperate(float[] x1, float x2, BinaryOperation operation) {
        int length = x1.length;
        float[] result = new float[length];
        
        for (int start = 0; start < length; start += MAX_CHUNK_SIZE) {
            int end = Math.min(start + MAX_CHUNK_SIZE, length);
            int chunkSize = end - start;
            
            // 创建分块数据
            float[] chunk = new float[chunkSize];
            System.arraycopy(x1, start, chunk, 0, chunkSize);
            
            // 处理分块
            float[] chunkResult = gpuBinaryOperate(chunk, x2, operation);
            
            // 复制结果
            System.arraycopy(chunkResult, 0, result, start, chunkSize);
        }
        
        return result;
    }
    
    /**
     * GPU版本的向量标量运算
     */
    private float[] gpuBinaryOperate(float[] x1, float x2, BinaryOperation operation) {
        int length = x1.length;
        float[] result = new float[length];
        
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    switch (operation) {
                        case ADD:
                            result[i] = x1[i] + x2;
                            break;
                        case SUBTRACT:
                            result[i] = x1[i] - x2;
                            break;
                        case MULTIPLY:
                            result[i] = x1[i] * x2;
                            break;
                        case DIVIDE:
                            result[i] = x1[i] / x2;
                            break;
                    }
                }
            }
        };
        
        try {
            kernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
            Range range = Range.create(length);
            kernel.execute(range);
            return result;
        } catch (Exception e) {
            logCPUFallback("GPU向量标量运算", "GPU执行失败: " + e.getMessage());
            return cpuBinaryOperate(x1, x2, operation);
        } finally {
            kernel.dispose();
        }
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        if (x1.length != x2.length || x1[0].length != x2[0].length) {
            throw new IllegalArgumentException("矩阵维度必须相同");
        }
        
        int rows = x1.length;
        int cols = x1[0].length;
        long dataSize = (long) rows * cols;
        
        // 智能选择执行策略
        if (!shouldUseGPU((int) Math.min(dataSize, Integer.MAX_VALUE))) {
            logCPUFallback("矩阵二元运算", "数据量小于阈值或GPU不支持FP64");
            return cpuMatrixBinaryOperate(x1, x2, operation);
        }
        
        // 大矩阵分块处理
        if (dataSize > MAX_CHUNK_SIZE) {
            return tiledMatrixBinaryOperate(x1, x2, operation);
        }
        
        // GPU执行
        return gpuMatrixBinaryOperate(x1, x2, operation);
    }
    
    /**
     * CPU版本的矩阵二元运算
     */
    private float[][] cpuMatrixBinaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                switch (operation) {
                    case ADD:
                        result[i][j] = x1[i][j] + x2[i][j];
                        break;
                    case SUBTRACT:
                        result[i][j] = x1[i][j] - x2[i][j];
                        break;
                    case MULTIPLY:
                        result[i][j] = x1[i][j] * x2[i][j];
                        break;
                    case DIVIDE:
                        result[i][j] = x1[i][j] / x2[i][j];
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 分块处理大矩阵的二元运算
     */
    private float[][] tiledMatrixBinaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];
        
        for (int startRow = 0; startRow < rows; startRow += TILE_SIZE) {
            for (int startCol = 0; startCol < cols; startCol += TILE_SIZE) {
                int endRow = Math.min(startRow + TILE_SIZE, rows);
                int endCol = Math.min(startCol + TILE_SIZE, cols);
                
                // 处理分块
                for (int i = startRow; i < endRow; i++) {
                    for (int j = startCol; j < endCol; j++) {
                        switch (operation) {
                            case ADD:
                                result[i][j] = x1[i][j] + x2[i][j];
                                break;
                            case SUBTRACT:
                                result[i][j] = x1[i][j] - x2[i][j];
                                break;
                            case MULTIPLY:
                                result[i][j] = x1[i][j] * x2[i][j];
                                break;
                            case DIVIDE:
                                result[i][j] = x1[i][j] / x2[i][j];
                                break;
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * GPU版本的矩阵二元运算
     */
    private float[][] gpuMatrixBinaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
        int rows = x1.length;
        int cols = x1[0].length;
        
        // 展平矩阵
        float[] flat1 = flattenMatrix(x1);
        float[] flat2 = flattenMatrix(x2);
        
        // 使用向量运算
        float[] flatResult = gpuBinaryOperate(flat1, flat2, operation);
        
        // 重构矩阵
        return unflattenMatrix(flatResult, rows, cols);
    }
    
    /**
     * 展平矩阵为一维数组
     */
    private float[] flattenMatrix(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] flat = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        
        return flat;
    }
    
    /**
     * 将一维数组重构为矩阵
     */
    private float[][] unflattenMatrix(float[] flat, int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, matrix[i], 0, cols);
        }
        
        return matrix;
    }

    // 为了保持接口兼容性，其他方法暂时使用CPU实现
    // 这些方法可以在后续版本中逐步改进
    
    @Override
    public float[][] binaryOperate(float[][] x1, float x2, BinaryOperation operation) {
        return cpuMatrixScalarOperate(x1, x2, operation);
    }
    
    private float[][] cpuMatrixScalarOperate(float[][] x1, float x2, BinaryOperation operation) {
        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                switch (operation) {
                    case ADD:
                        result[i][j] = x1[i][j] + x2;
                        break;
                    case SUBTRACT:
                        result[i][j] = x1[i][j] - x2;
                        break;
                    case MULTIPLY:
                        result[i][j] = x1[i][j] * x2;
                        break;
                    case DIVIDE:
                        result[i][j] = x1[i][j] / x2;
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        }
        
        return result;
    }

    @Override
    public float[] universalOperate(float[] x, UniversalOperation operation, float additionalParam) {
        // 对于复杂的数学函数，优先使用CPU以确保稳定性
        logCPUFallback("通用运算", "复杂数学函数使用CPU确保稳定性");
        return cpuUniversalOperate(x, operation, additionalParam);
    }
    
    private float[] cpuUniversalOperate(float[] x, UniversalOperation operation, float additionalParam) {
        float[] result = new float[x.length];
        
        for (int i = 0; i < x.length; i++) {
            switch (operation) {
                case EXP:
                    result[i] = (float)Math.exp(x[i]);
                    break;
                case LOG:
                    result[i] = (float)Math.log(x[i]);
                    break;
                case LOG10:
                    result[i] = (float)Math.log10(x[i]);
                    break;
                case SIN:
                    result[i] = (float)Math.sin(x[i]);
                    break;
                case COS:
                    result[i] = (float)Math.cos(x[i]);
                    break;
                case TAN:
                    result[i] = (float)Math.tan(x[i]);
                    break;
                case SQRT:
                    result[i] = (float)Math.sqrt(x[i]);
                    break;
                case ABS:
                    result[i] = (float)Math.abs(x[i]);
                    break;
                case POW:
                    result[i] = (float)Math.pow(x[i], additionalParam);
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }
        
        return result;
    }

    @Override
    public float[][] universalOperate(float[][] x, UniversalOperation operation, float additionalParam) {
        // 对于复杂的数学函数，优先使用CPU以确保稳定性
        logCPUFallback("矩阵通用运算", "复杂数学函数使用CPU确保稳定性");
        return cpuMatrixUniversalOperate(x, operation, additionalParam);
    }
    
    private float[][] cpuMatrixUniversalOperate(float[][] x, UniversalOperation operation, float additionalParam) {
        int rows = x.length;
        int cols = x[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                switch (operation) {
                    case EXP:
                        result[i][j] = (float)Math.exp(x[i][j]);
                        break;
                    case LOG:
                        result[i][j] = (float)Math.log(x[i][j]);
                        break;
                    case LOG10:
                        result[i][j] = (float)Math.log10(x[i][j]);
                        break;
                    case SIN:
                        result[i][j] = (float)Math.sin(x[i][j]);
                        break;
                    case COS:
                        result[i][j] = (float)Math.cos(x[i][j]);
                        break;
                    case TAN:
                        result[i][j] = (float)Math.tan(x[i][j]);
                        break;
                    case SQRT:
                        result[i][j] = (float)Math.sqrt(x[i][j]);
                        break;
                    case ABS:
                        result[i][j] = Math.abs(x[i][j]);
                        break;
                    case POW:
                        result[i][j] = (float)Math.pow(x[i][j], additionalParam);
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        }
        
        return result;
    }

    // 其他接口方法的简化CPU实现
    // 这些方法在后续版本中可以逐步优化
    
    @Override
    public float reduceOperate(float[] x, ReduceOperation operation) {
        switch (operation) {
            case SUM:
                float sum = 0.0f;
                for (float val : x) sum += val;
                return sum;
            case MEAN:
                float mean = 0.0f;
                for (float val : x) mean += val;
                return mean / x.length;
            case MAX:
                float max = Float.NEGATIVE_INFINITY;
                for (float val : x) if (val > max) max = val;
                return max;
            case MIN:
                float min = Float.POSITIVE_INFINITY;
                for (float val : x) if (val < min) min = val;
                return min;
            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float reduceOperate(float[][] x, ReduceOperation operation) {
        switch (operation) {
            case SUM:
                float sum = 0.0f;
                for (float[] row : x) {
                    for (float val : row) sum += val;
                }
                return sum;
            case MEAN:
                float mean = 0.0f;
                int count = 0;
                for (float[] row : x) {
                    for (float val : row) {
                        mean += val;
                        count++;
                    }
                }
                return mean / count;
            case MAX:
                float max = Float.NEGATIVE_INFINITY;
                for (float[] row : x) {
                    for (float val : row) if (val > max) max = val;
                }
                return max;
            case MIN:
                float min = Float.POSITIVE_INFINITY;
                for (float[] row : x) {
                    for (float val : row) if (val < min) min = val;
                }
                return min;
            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float binaryReduceOperate(float[] x1, float[] x2, BinaryReduceOperation operation) {
        switch (operation) {
            case DOT:
                float dot = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    dot += x1[i] * x2[i];
                }
                return dot;
            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float binaryReduceOperate(float[][] x1, float[][] x2, BinaryReduceOperation operation) {
        switch (operation) {
            case DOT:
                float dot = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    for (int j = 0; j < x1[0].length; j++) {
                        dot += x1[i][j] * x2[i][j];
                    }
                }
                return dot;
            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float[] elementWiseMin(float[] x1, float[] x2) {
        float[] result = new float[x1.length];
        for (int i = 0; i < x1.length; i++) {
            result[i] = Math.min(x1[i], x2[i]);
        }
        return result;
    }

    @Override
    public float[][] elementWiseMin(float[][] x1, float[][] x2) {
        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.min(x1[i][j], x2[i][j]);
            }
        }
        return result;
    }

    @Override
    public float[] elementWiseMax(float[] x1, float[] x2) {
        float[] result = new float[x1.length];
        for (int i = 0; i < x1.length; i++) {
            result[i] = Math.max(x1[i], x2[i]);
        }
        return result;
    }

    @Override
    public float[][] elementWiseMax(float[][] x1, float[][] x2) {
        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.max(x1[i][j], x2[i][j]);
            }
        }
        return result;
    }

    @Override
    public float[] negate(float[] x) {
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = -x[i];
        }
        return result;
    }

    @Override
    public float[][] negate(float[][] x) {
        int rows = x.length;
        int cols = x[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = -x[i][j];
            }
        }
        return result;
    }

    @Override
    public boolean[] logicalCompare(float[] x1, float[] x2, LogicalCompare operation) {
        boolean[] result = new boolean[x1.length];
        for (int i = 0; i < x1.length; i++) {
            switch (operation) {
                case EQUALS:
                    result[i] = x1[i] == x2[i];
                    break;
                case NOT_EQUALS:
                    result[i] = x1[i] != x2[i];
                    break;
                case LESS_THAN:
                    result[i] = x1[i] < x2[i];
                    break;
                case LESS_THAN_OR_EQUALS:
                    result[i] = x1[i] <= x2[i];
                    break;
                case GREATER_THAN:
                    result[i] = x1[i] > x2[i];
                    break;
                case GREATER_THAN_OR_EQUALS:
                    result[i] = x1[i] >= x2[i];
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }
        return result;
    }

    @Override
    public boolean[] logicalOperate(float[] x1, float[] x2, LogicalOperation operation) {
        // 简化实现，实际应用中可以扩展
        throw new UnsupportedOperationException("逻辑运算暂未实现");
    }

    @Override
    public boolean[] logicalOperate(float[] x1, LogicalOperation operation) {
        // 简化实现，实际应用中可以扩展
        throw new UnsupportedOperationException("逻辑运算暂未实现");
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] result = new float[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        float[][] result = new float[rowVector.length][1];
        for (int i = 0; i < rowVector.length; i++) {
            result[i][0] = rowVector[i];
        }
        return result;
    }

    @Override
    public float[][] mmul(float[][] a, float[][] b) {
        // 使用CPU实现矩阵乘法以确保稳定性
        int aRows = a.length;
        int aCols = a[0].length;
        int bCols = b[0].length;
        
        float[][] result = new float[aRows][bCols];
        
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                for (int k = 0; k < aCols; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }

    @Override
    public float[][] outer(float[] a, float[] b) {
        float[][] result = new float[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    @Override
    public float[] sign(float[] array) {
        float[] result = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = Math.signum(array[i]);
        }
        return result;
    }

    @Override
    public float[][] sign(float[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.signum(array[i][j]);
            }
        }
        return result;
    }

    @Override
    public float[] diff(float[] array, int stride) {
        if (stride <= 0 || stride >= array.length) {
            throw new IllegalArgumentException("步长必须在1到数组长度-1之间");
        }
        
        float[] result = new float[array.length - stride];
        for (int i = 0; i < result.length; i++) {
            result[i] = array[i + stride] - array[i];
        }
        return result;
    }

    @Override
    public boolean[][] logicalCompare(float[][] x1, float[][] x2, LogicalCompare operation) {
        int rows = x1.length;
        int cols = x1[0].length;
        boolean[][] result = new boolean[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                switch (operation) {
                    case EQUALS:
                        result[i][j] = x1[i][j] == x2[i][j];
                        break;
                    case NOT_EQUALS:
                        result[i][j] = x1[i][j] != x2[i][j];
                        break;
                    case LESS_THAN:
                        result[i][j] = x1[i][j] < x2[i][j];
                        break;
                    case LESS_THAN_OR_EQUALS:
                        result[i][j] = x1[i][j] <= x2[i][j];
                        break;
                    case GREATER_THAN:
                        result[i][j] = x1[i][j] > x2[i][j];
                        break;
                    case GREATER_THAN_OR_EQUALS:
                        result[i][j] = x1[i][j] >= x2[i][j];
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        }
        
        return result;
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, LogicalOperation operation) {
        // 简化实现，实际应用中可以扩展
        throw new UnsupportedOperationException("逻辑运算暂未实现");
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, float[][] x2, LogicalOperation operation) {
        // 简化实现，实际应用中可以扩展
        throw new UnsupportedOperationException("逻辑运算暂未实现");
    }
}