package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import com.yishape.lab.util.Tuple2;

/**
 * 时间序列数据类 / Time Series Data Class
 * <p>
 * 表示时间序列数据，包含时间戳和对应的数值数据。
 * 支持单变量和多变量时间序列，提供基本的数据操作和访问方法。
 * </p>
 * <p>
 * Represents time series data containing timestamps and corresponding numerical data.
 * Supports univariate and multivariate time series, provides basic data operations and access methods.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesData {
    
    private final List<LocalDateTime> timestamps;
    private final IMatrix<Double> data;
    private final String[] columnNames;
    private final double samplingRate;
    
    /**
     * 构造函数 - 从时间戳和向量数据创建 / Constructor - Create from timestamps and vector data
     *
     * @param timestamps 时间戳列表 / List of timestamps
     * @param values 数值向量 / Values vector
     * @param columnName 列名 / Column name
     */
    public TimeSeriesData(List<LocalDateTime> timestamps, IVector<Double> values, String columnName) {
        this.timestamps = new ArrayList<>(timestamps);
        this.columnNames = new String[]{columnName};
        this.data = Linalg.matrix(new double[][]{values.toDoubleArray()});
        this.samplingRate = calculateSamplingRate();
    }
    
    /**
     * 构造函数 - 从时间戳和矩阵数据创建 / Constructor - Create from timestamps and matrix data
     *
     * @param timestamps 时间戳列表 / List of timestamps
     * @param data 数据矩阵 / Data matrix
     * @param columnNames 列名数组 / Column names array
     */
    public TimeSeriesData(List<LocalDateTime> timestamps, IMatrix<Double> data, String[] columnNames) {
        this.timestamps = new ArrayList<>(timestamps);
        this.data = data.copy();
        this.columnNames = columnNames.clone();
        this.samplingRate = calculateSamplingRate();
    }
    
    /**
     * 构造函数 - 从向量和采样率创建 / Constructor - Create from vector and sampling rate
     *
     * @param values 数值向量 / Values vector
     * @param samplingRate 采样率 (Hz) / Sampling rate (Hz)
     * @param columnName 列名 / Column name
     * @param startTime 开始时间 / Start time
     */
    public TimeSeriesData(IVector<Double> values, double samplingRate, String columnName, LocalDateTime startTime) {
        this.timestamps = generateTimestamps(values.length(), samplingRate, startTime);
        this.columnNames = new String[]{columnName};
        this.data = Linalg.matrix(new double[][]{values.toDoubleArray()});
        this.samplingRate = samplingRate;
    }
    
    /**
     * 构造函数 - 从数组创建单变量时间序列 / Constructor - Create univariate time series from arrays
     *
     * @param timestamps 时间戳数组 / Timestamp array
     * @param values 数值数组 / Values array
     * @param columnName 列名 / Column name
     */
    public TimeSeriesData(LocalDateTime[] timestamps, double[] values, String columnName) {
        this.timestamps = Arrays.asList(timestamps);
        this.columnNames = new String[]{columnName};
        this.data = Linalg.matrix(new double[][]{values});
        this.samplingRate = calculateSamplingRate();
    }
    
    /**
     * 构造函数 - 从数组创建多变量时间序列 / Constructor - Create multivariate time series from arrays
     *
     * @param timestamps 时间戳数组 / Timestamp array
     * @param data 数据二维数组 / Data 2D array
     * @param columnNames 列名数组 / Column names array
     */
    public TimeSeriesData(LocalDateTime[] timestamps, double[][] data, String[] columnNames) {
        this.timestamps = Arrays.asList(timestamps);
        this.columnNames = columnNames.clone();
        this.data = Linalg.matrix(data);
        this.samplingRate = calculateSamplingRate();
    }
    
    /**
     * 获取时间戳列表 / Get timestamps list
     *
     * @return 时间戳列表 / Timestamps list
     */
    public List<LocalDateTime> getTimestamps() {
        return new ArrayList<>(timestamps);
    }
    
    /**
     * 获取时间戳数组 / Get timestamps array
     *
     * @return 时间戳数组 / Timestamps array
     */
    public LocalDateTime[] getTimestampsArray() {
        return timestamps.toArray(new LocalDateTime[0]);
    }
    
    /**
     * 获取数据矩阵 / Get data matrix
     *
     * @return 数据矩阵 / Data matrix
     */
    public IMatrix<Double> getData() {
        return data.copy();
    }
    
    /**
     * 获取列名数组 / Get column names array
     *
     * @return 列名数组 / Column names array
     */
    public String[] getColumnNames() {
        return columnNames.clone();
    }
    
    /**
     * 获取采样率 / Get sampling rate
     *
     * @return 采样率 (Hz) / Sampling rate (Hz)
     */
    public double getSamplingRate() {
        return samplingRate;
    }
    
    /**
     * 获取时间序列长度 / Get time series length
     *
     * @return 时间序列长度 / Time series length
     */
    public int getLength() {
        return timestamps.size();
    }
    
    /**
     * 获取变量数量 / Get number of variables
     *
     * @return 变量数量 / Number of variables
     */
    public int getNumVariables() {
        return data.getColNum();
    }
    
    /**
     * 是否为单变量时间序列 / Check if univariate time series
     *
     * @return 是否为单变量 / Whether univariate
     */
    public boolean isUnivariate() {
        return getNumVariables() == 1;
    }
    
    /**
     * 是否为多变量时间序列 / Check if multivariate time序列
     *
     * @return 是否为多变量 / Whether multivariate
     */
    public boolean isMultivariate() {
        return getNumVariables() > 1;
    }
    
    /**
     * 获取指定变量的数据 / Get data for specified variable
     *
     * @param variableIndex 变量索引 / Variable index
     * @return 变量数据向量 / Variable data vector
     */
    public IVector<Double> getVariable(int variableIndex) {
        if (variableIndex < 0 || variableIndex >= getNumVariables()) {
            throw new IllegalArgumentException("变量索引超出范围");
        }
        return data.getColumn(variableIndex);
    }
    
    /**
     * 获取指定变量名的数据 / Get data for specified variable name
     *
     * @param columnName 列名 / Column name
     * @return 变量数据向量 / Variable data vector
     */
    public IVector<Double> getVariable(String columnName) {
        int index = getVariableIndex(columnName);
        return getVariable(index);
    }
    
    /**
     * 获取变量索引 / Get variable index
     *
     * @param columnName 列名 / Column name
     * @return 变量索引 / Variable index
     */
    public int getVariableIndex(String columnName) {
        for (int i = 0; i < columnNames.length; i++) {
            if (columnNames[i].equals(columnName)) {
                return i;
            }
        }
        throw new IllegalArgumentException("未找到变量: " + columnName);
    }
    
    /**
     * 获取时间范围 / Get time range
     *
     * @return 时间范围元组 (开始时间, 结束时间) / Time range tuple (start time, end time)
     */
    public LocalDateTime[] getTimeRange() {
        return new LocalDateTime[]{timestamps.get(0), timestamps.get(timestamps.size() - 1)};
    }
    
    /**
     * 获取时间间隔 / Get time interval
     *
     * @return 时间间隔 (秒) / Time interval (seconds)
     */
    public double getTimeInterval() {
        if (timestamps.size() < 2) {
            return 0.0;
        }
        return ChronoUnit.MILLIS.between(timestamps.get(0), timestamps.get(1)) / 1000.0;
    }
    
    /**
     * 切片时间序列 / Slice time series
     *
     * @param startIndex 开始索引 / Start index
     * @param endIndex 结束索引 / End index
     * @return 切片后的时间序列 / Sliced time series
     */
    public TimeSeriesData slice(int startIndex, int endIndex) {
        if (startIndex < 0 || endIndex > getLength() || startIndex >= endIndex) {
            throw new IllegalArgumentException("无效的切片索引");
        }
        
        List<LocalDateTime> slicedTimestamps = timestamps.subList(startIndex, endIndex);
        IMatrix<Double> slicedData = data.sliceRows(startIndex, endIndex);
        
        return new TimeSeriesData(slicedTimestamps, slicedData, columnNames);
    }
    
    /**
     * 按时间范围切片 / Slice by time range
     *
     * @param startTime 开始时间 / Start time
     * @param endTime 结束时间 / End time
     * @return 切片后的时间序列 / Sliced time series
     */
    public TimeSeriesData slice(LocalDateTime startTime, LocalDateTime endTime) {
        int startIndex = findTimeIndex(startTime);
        int endIndex = findTimeIndex(endTime);
        
        if (startIndex == -1 || endIndex == -1 || startIndex >= endIndex) {
            throw new IllegalArgumentException("无效的时间范围");
        }
        
        return slice(startIndex, endIndex + 1);
    }
    
    /**
     * 重采样时间序列 / Resample time series
     *
     * @param newSamplingRate 新采样率 / New sampling rate
     * @return 重采样后的时间序列 / Resampled time series
     */
    public TimeSeriesData resample(double newSamplingRate) {
        if (newSamplingRate <= 0) {
            throw new IllegalArgumentException("采样率必须为正数");
        }
        
        double ratio = samplingRate / newSamplingRate;
        int newLength = (int) Math.round(getLength() / ratio);
        
        IVector<Double> newIndices = Linalg.range(newLength).multiplyScalar(ratio);
        IMatrix<Double> newData = Linalg.zeros(newLength, getNumVariables());
        
        for (int i = 0; i < getNumVariables(); i++) {
            IVector<Double> variable = getVariable(i);
            IVector<Double> resampledVariable = interpolate(variable, newIndices);
            newData.setColumn(i, resampledVariable);
        }
        
        List<LocalDateTime> newTimestamps = generateTimestamps(newLength, newSamplingRate, timestamps.get(0));
        
        return new TimeSeriesData(newTimestamps, newData, columnNames);
    }
    
    /**
     * 添加噪声 / Add noise
     *
     * @param noiseLevel 噪声水平 / Noise level
     * @return 添加噪声后的时间序列 / Time series with added noise
     */
    public TimeSeriesData addNoise(double noiseLevel) {
        IMatrix<Double> noisyData = data.copy();
        
        for (int i = 0; i < getNumVariables(); i++) {
            IVector<Double> variable = getVariable(i);
            IVector<Double> noise = Linalg.randn(getLength()).multiplyScalar(noiseLevel);
            IVector<Double> noisyVariable = variable.add(noise);
            noisyData.setColumn(i, noisyVariable);
        }
        
        return new TimeSeriesData(timestamps, noisyData, columnNames);
    }
    
    /**
     * 转换为单变量时间序列 / Convert to univariate time series
     *
     * @param variableIndex 变量索引 / Variable index
     * @return 单变量时间序列 / Univariate time series
     */
    public TimeSeriesData toUnivariate(int variableIndex) {
        IVector<Double> variable = getVariable(variableIndex);
        return new TimeSeriesData(timestamps, variable, columnNames[variableIndex]);
    }
    
    /**
     * 转换为单变量时间序列 / Convert to univariate time series
     *
     * @param columnName 列名 / Column name
     * @return 单变量时间序列 / Univariate time series
     */
    public TimeSeriesData toUnivariate(String columnName) {
        int index = getVariableIndex(columnName);
        return toUnivariate(index);
    }
    
    /**
     * 计算基本统计信息 / Calculate basic statistics
     *
     * @return 统计信息矩阵 / Statistics matrix
     */
    public IMatrix<Double> getStatistics() {
        IMatrix<Double> stats = Linalg.zeros(6, getNumVariables());
        
        for (int i = 0; i < getNumVariables(); i++) {
            IVector<Double> variable = getVariable(i);
            stats.set(0, i, variable.mean());
            stats.set(1, i, variable.std());
            stats.set(2, i, variable.min());
            stats.set(3, i, variable.max());
            stats.set(4, i, variable.median());
            stats.set(5, i, variable.var());
        }
        
        return stats;
    }
    
    /**
     * 转换为双数组格式 / Convert to double array format
     *
     * @return 包含时间戳和数据的双数组 / Double arrays containing timestamps and data
     */
    public Tuple2<double[], double[][]> toDoubleArrays() {
        // Convert timestamps to double array (seconds since epoch)
        double[] timeArray = new double[timestamps.size()];
        for (int i = 0; i < timestamps.size(); i++) {
            timeArray[i] = timestamps.get(i).atZone(java.time.ZoneId.systemDefault()).toEpochSecond();
        }
        
        // Convert data to double array
        double[][] dataArray = data.toDoubleArray();
        
        return new Tuple2<>(timeArray, dataArray);
    }
    
    /**
     * 合并时间序列 / Merge time series
     *
     * @param other 另一个时间序列 / Another time series
     * @param newName 新列名 / New column name
     * @return 合并后的时间序列 / Merged time series
     */
    public TimeSeriesData merge(TimeSeriesData other, String newName) {
        // For simplicity, we'll assume same timestamps and merge as multivariate
        String[] newColumnNames = new String[columnNames.length + other.columnNames.length];
        System.arraycopy(columnNames, 0, newColumnNames, 0, columnNames.length);
        System.arraycopy(other.columnNames, 0, newColumnNames, columnNames.length, other.columnNames.length);
        
        IMatrix<Double> newData = data.hstack(other.data);
        
        return new TimeSeriesData(timestamps, newData, newColumnNames);
    }
    
    /**
     * 标准化时间序列 / Normalize time series
     *
     * @return 标准化后的时间序列 / Normalized time series
     */
    public TimeSeriesData normalize() {
        IMatrix<Double> normalizedData = data.copy();
        
        for (int i = 0; i < getNumVariables(); i++) {
            IVector<Double> variable = getVariable(i);
            double mean = variable.mean();
            double std = variable.std();
            if (std > 0) {
                IVector<Double> normalizedVariable = variable.subScalar(mean).divideByScalar(std);
                normalizedData.setColumn(i, normalizedVariable);
            }
        }
        
        return new TimeSeriesData(timestamps, normalizedData, columnNames);
    }
    
    /**
     * 移动窗口操作 / Moving window operation
     *
     * @param windowSize 窗口大小 / Window size
     * @param operation 操作函数 / Operation function
     * @return 处理后的时间序列 / Processed time series
     */
    public TimeSeriesData movingWindow(int windowSize, java.util.function.Function<IVector<Double>, Double> operation) {
        if (windowSize <= 0 || windowSize > getLength()) {
            throw new IllegalArgumentException("无效的窗口大小");
        }
        
        int resultLength = getLength() - windowSize + 1;
        double[] result = new double[resultLength];
        
        for (int i = 0; i < resultLength; i++) {
            IVector<Double> window = data.sliceRows(i, i + windowSize).getColumn(0); // Assuming univariate for simplicity
            result[i] = operation.apply(window);
        }
        
        LocalDateTime[] newTimestamps = new LocalDateTime[resultLength];
        for (int i = 0; i < resultLength; i++) {
            newTimestamps[i] = timestamps.get(i + windowSize - 1);
        }
        
        return new TimeSeriesData(newTimestamps, result, columnNames[0] + "_windowed");
    }
    
    @Override
    public String toString() {
        return String.format("TimeSeriesData{length=%d, variables=%d, samplingRate=%.2fHz, columns=%s}",
                getLength(), getNumVariables(), samplingRate, String.join(",", columnNames));
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        TimeSeriesData that = (TimeSeriesData) obj;
        
        if (Double.compare(that.samplingRate, samplingRate) != 0) return false;
        if (!timestamps.equals(that.timestamps)) return false;
        if (!data.equals(that.data)) return false;
        return Arrays.equals(columnNames, that.columnNames);
    }
    
    @Override
    public int hashCode() {
        int result;
        long temp;
        result = timestamps.hashCode();
        result = 31 * result + data.hashCode();
        result = 31 * result + Arrays.hashCode(columnNames);
        temp = Double.doubleToLongBits(samplingRate);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
    
    // ========== 静态工厂方法 / Static Factory Methods ==========
    
    /**
     * 创建时间序列数据对象 / Create time series data object
     *
     * @param values 数值向量 / Values vector
     * @param name 列名 / Column name
     * @return 时间序列数据对象 / Time series data object
     */
    public static TimeSeriesData of(IVector<Double> values, String name) {
        LocalDateTime startTime = LocalDateTime.now();
        return new TimeSeriesData(values, 1.0, name, startTime);
    }
    
    /**
     * 创建时间序列数据对象 / Create time series data object
     *
     * @param values 数值向量 / Values vector
     * @param samplingRate 采样率 / Sampling rate
     * @param name 列名 / Column name
     * @param startTime 开始时间 / Start time
     * @return 时间序列数据对象 / Time series data object
     */
    public static TimeSeriesData of(IVector<Double> values, double samplingRate, String name, LocalDateTime startTime) {
        return new TimeSeriesData(values, samplingRate, name, startTime);
    }
    
    /**
     * 创建时间序列数据对象 / Create time series data object
     *
     * @param timestamps 时间戳数组 / Timestamp array
     * @param values 数值数组 / Values array
     * @param name 列名 / Column name
     * @return 时间序列数据对象 / Time series data object
     */
    public static TimeSeriesData of(LocalDateTime[] timestamps, double[] values, String name) {
        return new TimeSeriesData(timestamps, values, name);
    }
    
    /**
     * 创建多变量时间序列数据对象 / Create multivariate time series data object
     *
     * @param timestamps 时间戳数组 / Timestamp array
     * @param data 数据二维数组 / Data 2D array
     * @param names 列名数组 / Column names array
     * @return 时间序列数据对象 / Time series data object
     */
    public static TimeSeriesData of(LocalDateTime[] timestamps, double[][] data, String[] names) {
        return new TimeSeriesData(timestamps, data, names);
    }
    
    /**
     * 创建时间序列数据对象 / Create time series data object
     *
     * @param values 数值向量 / Values vector
     * @param name 列名 / Column name
     * @param frequency 时间间隔 (秒) / Time interval (seconds)
     * @return 时间序列数据对象 / Time series data object
     */
    public static TimeSeriesData of(IVector<Double> values, String name, double frequency) {
        LocalDateTime startTime = LocalDateTime.now();
        return new TimeSeriesData(values, 1.0/frequency, name, startTime);
    }
    
    /**
     * 创建时间序列数据对象（从双数组） / Create time series data object (from double arrays)
     *
     * @param timeArray 时间戳数组 (秒 since epoch) / Timestamp array (seconds since epoch)
     * @param dataArray 数据数组 / Data array
     * @param names 列名数组 / Column names array
     * @return 时间序列数据对象 / Time series data object
     */
    public static TimeSeriesData of(double[] timeArray, double[][] dataArray, String[] names) {
        LocalDateTime[] timestamps = new LocalDateTime[timeArray.length];
        for (int i = 0; i < timeArray.length; i++) {
            timestamps[i] = LocalDateTime.ofEpochSecond((long) timeArray[i], 0, java.time.ZoneOffset.UTC);
        }
        return new TimeSeriesData(timestamps, dataArray, names);
    }
    
    /**
     * 创建示例时间序列数据 / Create sample time series data
     *
     * @param length 长度 / Length
     * @param name 名称 / Name
     * @return 示例时间序列数据 / Sample time series data
     */
    public static TimeSeriesData sample(int length, String name) {
        IVector<Double> data = Linalg.randn(length);
        return TimeSeriesData.of(data, name);
    }
    
    /**
     * 创建正弦波时间序列 / Create sine wave time series
     *
     * @param length 长度 / Length
     * @param frequency 频率 / Frequency
     * @param name 名称 / Name
     * @return 正弦波时间序列 / Sine wave time series
     */
    public static TimeSeriesData sineWave(int length, double frequency, String name) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = Math.sin(2 * Math.PI * frequency * i / length);
        }
        return TimeSeriesData.of(Linalg.vector(values), name);
    }
    
    // ========== 构建器模式 / Builder Pattern ==========
    
    /**
     * 创建构建器 / Create builder
     *
     * @return 时间序列数据构建器 / Time series data builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * 时间序列数据构建器 / Time series data builder
     */
    public static class Builder {
        private List<LocalDateTime> timestamps;
        private IMatrix<Double> data;
        private String[] columnNames;
        private double samplingRate = 1.0;
        private LocalDateTime startTime = LocalDateTime.now();
        
        public Builder timestamps(LocalDateTime[] timestamps) {
            this.timestamps = Arrays.asList(timestamps);
            return this;
        }
        
        public Builder timestamps(List<LocalDateTime> timestamps) {
            this.timestamps = new ArrayList<>(timestamps);
            return this;
        }
        
        public Builder data(IVector<Double> data, String columnName) {
            this.data = Linalg.matrix(new double[][]{data.toDoubleArray()});
            this.columnNames = new String[]{columnName};
            return this;
        }
        
        public Builder data(IMatrix<Double> data, String[] columnNames) {
            this.data = data.copy();
            this.columnNames = columnNames.clone();
            return this;
        }
        
        public Builder data(double[] data, String columnName) {
            this.data = Linalg.matrix(new double[][]{data});
            this.columnNames = new String[]{columnName};
            return this;
        }
        
        public Builder data(double[][] data, String[] columnNames) {
            this.data = Linalg.matrix(data);
            this.columnNames = columnNames.clone();
            return this;
        }
        
        public Builder samplingRate(double samplingRate) {
            this.samplingRate = samplingRate;
            return this;
        }
        
        public Builder startTime(LocalDateTime startTime) {
            this.startTime = startTime;
            return this;
        }
        
        public Builder frequency(double frequency) {
            this.samplingRate = frequency;
            return this;
        }
        
        public TimeSeriesData build() {
            if (data == null) {
                throw new IllegalStateException("数据不能为空 / Data cannot be null");
            }
            
            if (timestamps == null) {
                // Generate timestamps based on data length and sampling rate
                timestamps = TimeSeriesData.generateTimestamps(data.rows(), samplingRate, startTime);
            }
            
            if (columnNames == null) {
                // Generate default column names
                columnNames = new String[data.getColNum()];
                for (int i = 0; i < columnNames.length; i++) {
                    columnNames[i] = "Variable" + (i + 1);
                }
            }
            
            return new TimeSeriesData(timestamps, data, columnNames);
        }
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 计算采样率 / Calculate sampling rate
     */
    private double calculateSamplingRate() {
        if (timestamps.size() < 2) {
            return 0.0;
        }
        
        double totalSeconds = ChronoUnit.MILLIS.between(timestamps.get(0), timestamps.get(timestamps.size() - 1)) / 1000.0;
        return (timestamps.size() - 1) / totalSeconds;
    }
    
    /**
     * 生成时间戳 / Generate timestamps
     */
    private static List<LocalDateTime> generateTimestamps(int length, double samplingRate, LocalDateTime startTime) {
        List<LocalDateTime> timestamps = new ArrayList<>();
        double intervalSeconds = 1.0 / samplingRate;
        
        for (int i = 0; i < length; i++) {
            LocalDateTime timestamp = startTime.plusNanos((long) (i * intervalSeconds * 1_000_000_000));
            timestamps.add(timestamp);
        }
        
        return timestamps;
    }
    
    /**
     * 查找时间索引 / Find time index
     */
    private int findTimeIndex(LocalDateTime time) {
        for (int i = 0; i < timestamps.size(); i++) {
            if (!timestamps.get(i).isBefore(time)) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * 线性插值 / Linear interpolation
     */
    private static IVector<Double> interpolate(IVector<Double> data, IVector<Double> indices) {
        int length = indices.length();
        IVector<Double> result = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            double index = indices.get(i);
            int lower = (int) Math.floor(index);
            int upper = (int) Math.ceil(index);
            
            if (lower < 0) {
                result.set(i, data.get(0));
            } else if (upper >= data.length()) {
                result.set(i, data.get(data.length() - 1));
            } else if (lower == upper) {
                result.set(i, data.get(lower));
            } else {
                double weight = index - lower;
                double value = data.get(lower) * (1 - weight) + data.get(upper) * weight;
                result.set(i, value);
            }
        }
        
        return result;
    }
}