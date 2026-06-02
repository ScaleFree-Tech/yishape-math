package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.Linalg;

import java.time.LocalDateTime;

/**
 * 时间序列数据类可用性测试 / Time Series Data Usability Test
 * <p>
 * This class demonstrates the enhanced usability features of the TimeSeriesData class.
 * </p>
 */
public class TimeSeriesDataUsabilityTest {
    
    public static void main(String[] args) {
        // 创建示例数据 / Create sample data
        System.out.println("=== 时间序列数据类可用性演示 ===");
        
        // 1. 使用静态工厂方法创建时间序列 / Create time series using static factory methods
        System.out.println("\n1. 使用静态工厂方法创建时间序列:");
        TimeSeriesData ts1 = TimeSeriesData.sample(100, "sample_data");
        System.out.println("创建的示例时间序列: " + ts1);
        
        // 2. 使用静态工厂创建时间序列 / Create time series using static factory
        System.out.println("\n2. 使用静态工厂创建时间序列:");
        TimeSeriesData ts2 = TimeSeriesData.of(Linalg.randn(50), "random_data");
        System.out.println("使用静态工厂创建的时间序列: " + ts2);
        
        // 3. 使用便捷的静态方法创建特殊时间序列 / Create special time series using convenience methods
        System.out.println("\n3. 创建特殊时间序列:");
        TimeSeriesData sineWave = TimeSeriesData.sineWave(100, 2.0, "sine_wave");
        System.out.println("创建的正弦波时间序列: " + sineWave);
        
        // 4. 数据操作演示 / Data operation demonstration
        System.out.println("\n4. 数据操作演示:");
        // 标准化 / Normalize
        TimeSeriesData normalized = ts1.normalize();
        System.out.println("标准化后的时间序列: " + normalized);
        
        // 添加噪声 / Add noise
        TimeSeriesData noisy = ts1.addNoise(0.1);
        System.out.println("添加噪声后的时间序列: " + noisy);
        
        // 5. 数据访问演示 / Data access demonstration
        System.out.println("\n5. 数据访问演示:");
        System.out.println("时间序列长度: " + ts1.getLength());
        System.out.println("变量数量: " + ts1.getNumVariables());
        System.out.println("是否为单变量: " + ts1.isUnivariate());
        System.out.println("采样率: " + ts1.getSamplingRate());
        
        // 6. 合并时间序列 / Merge time series
        System.out.println("\n6. 合并时间序列:");
        TimeSeriesData merged = ts1.merge(sineWave, "merged");
        System.out.println("合并后的时间序列: " + merged);
        
        System.out.println("\n=== 演示完成 ===");
    }
}