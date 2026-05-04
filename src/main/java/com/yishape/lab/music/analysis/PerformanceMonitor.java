package com.yishape.lab.music.analysis;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.time.Instant;
import java.time.Duration;
import java.util.Map;
import java.util.HashMap;

/**
 * 性能监控器 / Performance Monitor
 * <p>
 * 监控音频分析的性能指标，包括执行时间、吞吐量、错误率等。
 * Monitor performance metrics of audio analysis, including execution time, throughput, error rate, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class PerformanceMonitor {
    
    /** 单例实例 / Singleton instance */
    private static final PerformanceMonitor INSTANCE = new PerformanceMonitor();
    
    /** 分析计数器 / Analysis counter */
    private final AtomicLong analysisCounter = new AtomicLong(0);
    
    /** 错误计数器 / Error counter */
    private final AtomicLong errorCounter = new AtomicLong(0);
    
    /** 总执行时间（纳秒）/ Total execution time (nanoseconds) */
    private final AtomicLong totalExecutionTime = new AtomicLong(0);
    
    /** 最小执行时间（纳秒）/ Minimum execution time (nanoseconds) */
    private final AtomicLong minExecutionTime = new AtomicLong(Long.MAX_VALUE);
    
    /** 最大执行时间（纳秒）/ Maximum execution time (nanoseconds) */
    private final AtomicLong maxExecutionTime = new AtomicLong(0);
    
    /** 算法统计信息 / Algorithm statistics */
    private final ConcurrentHashMap<String, AlgorithmStats> algorithmStats = new ConcurrentHashMap<>();
    
    /** 开始时间 / Start time */
    private final Instant startTime = Instant.now();
    
    /**
     * 算法统计信息类 / Algorithm statistics class
     */
    private static class AlgorithmStats {
        final AtomicLong count = new AtomicLong(0);
        final AtomicLong totalTime = new AtomicLong(0);
        final AtomicLong minTime = new AtomicLong(Long.MAX_VALUE);
        final AtomicLong maxTime = new AtomicLong(0);
        
        void recordExecution(long executionTime) {
            count.incrementAndGet();
            totalTime.addAndGet(executionTime);
            
            // 更新最小值 / Update minimum
            long currentMin;
            do {
                currentMin = minTime.get();
            } while (executionTime < currentMin && !minTime.compareAndSet(currentMin, executionTime));
            
            // 更新最大值 / Update maximum
            long currentMax;
            do {
                currentMax = maxTime.get();
            } while (executionTime > currentMax && !maxTime.compareAndSet(currentMax, executionTime));
        }
        
        double getAverageTime() {
            long total = totalTime.get();
            long cnt = count.get();
            return cnt > 0 ? (double) total / cnt : 0.0;
        }
    }
    
    /**
     * 私有构造函数 / Private constructor
     */
    private PerformanceMonitor() {
    }
    
    /**
     * 获取单例实例 / Get singleton instance
     *
     * @return PerformanceMonitor单例实例 / Singleton instance of PerformanceMonitor
     */
    public static PerformanceMonitor getInstance() {
        return INSTANCE;
    }
    
    /**
     * 记录分析开始 / Record analysis start
     * <p>
     * 开始跟踪一个分析操作的执行时间。
     * Starts tracking the execution time of an analysis operation.
     * </p>
     *
     * @param algorithmName 算法名称 / Algorithm name
     * @return 分析上下文，用于记录完成 / Analysis context for recording completion
     */
    public AnalysisContext recordAnalysisStart(String algorithmName) {
        return new AnalysisContext(algorithmName);
    }
    
    /**
     * 记录分析完成 / Record analysis completion
     * <p>
     * 记录分析操作的完成，更新性能统计信息。
     * Records the completion of an analysis operation and updates performance statistics.
     * </p>
     *
     * @param context 分析上下文（由recordAnalysisStart返回）/ Analysis context (returned by recordAnalysisStart)
     * @param success 分析是否成功 / Whether analysis was successful
     */
    public void recordAnalysisComplete(AnalysisContext context, boolean success) {
        if (context == null) {
            return;
        }
        
        long executionTime = context.getExecutionTime();
        
        // 更新全局统计 / Update global statistics
        analysisCounter.incrementAndGet();
        totalExecutionTime.addAndGet(executionTime);
        
        // 更新最小值 / Update minimum
        long currentMin;
        do {
            currentMin = minExecutionTime.get();
        } while (executionTime < currentMin && !minExecutionTime.compareAndSet(currentMin, executionTime));
        
        // 更新最大值 / Update maximum
        long currentMax;
        do {
            currentMax = maxExecutionTime.get();
        } while (executionTime > currentMax && !maxExecutionTime.compareAndSet(currentMax, executionTime));
        
        // 更新错误计数 / Update error count
        if (!success) {
            errorCounter.incrementAndGet();
        }
        
        // 更新算法统计 / Update algorithm statistics
        if (context.getAlgorithmName() != null) {
            AlgorithmStats stats = algorithmStats.computeIfAbsent(
                context.getAlgorithmName(), k -> new AlgorithmStats()
            );
            stats.recordExecution(executionTime);
        }
    }
    
    /**
     * 获取性能统计 / Get performance statistics
     * <p>
     * 返回所有收集到的性能统计信息。
     * Returns all collected performance statistics.
     * </p>
     *
     * @return 性能统计对象 / Performance statistics object
     */
    public PerformanceStatistics getStatistics() {
        long totalCount = analysisCounter.get();
        long totalTime = totalExecutionTime.get();
        long minTime = minExecutionTime.get() == Long.MAX_VALUE ? 0 : minExecutionTime.get();
        long maxTime = maxExecutionTime.get();
        long errorCount = errorCounter.get();
        
        double avgTime = totalCount > 0 ? (double) totalTime / totalCount : 0.0;
        double errorRate = totalCount > 0 ? (double) errorCount / totalCount * 100.0 : 0.0;
        double throughput = getUptimeSeconds() > 0 ? (double) totalCount / getUptimeSeconds() : 0.0;
        
        // 创建算法统计副本 / Create algorithm statistics copy
        Map<String, AlgorithmStatistics> algoStats = new HashMap<>();
        for (Map.Entry<String, AlgorithmStats> entry : algorithmStats.entrySet()) {
            AlgorithmStats stats = entry.getValue();
            algoStats.put(entry.getKey(), new AlgorithmStatistics(
                stats.count.get(),
                stats.getAverageTime(),
                stats.minTime.get() == Long.MAX_VALUE ? 0 : stats.minTime.get(),
                stats.maxTime.get(),
                stats.totalTime.get()
            ));
        }
        
        return new PerformanceStatistics(
            totalCount, totalTime, avgTime, minTime, maxTime,
            errorCount, errorRate, throughput, getUptimeSeconds(), algoStats
        );
    }
    
    /**
     * 重置统计信息 / Reset statistics
     * <p>
     * 清空所有性能统计计数器。
     * Clears all performance statistics counters.
     */
    public void reset() {
        analysisCounter.set(0);
        errorCounter.set(0);
        totalExecutionTime.set(0);
        minExecutionTime.set(Long.MAX_VALUE);
        maxExecutionTime.set(0);
        algorithmStats.clear();
    }
    
    /**
     * 获取运行时间（秒）/ Get uptime (seconds)
     */
    private double getUptimeSeconds() {
        return Duration.between(startTime, Instant.now()).getSeconds();
    }
    
    /**
     * 分析上下文 / Analysis context
     */
    public static class AnalysisContext {
        private final String algorithmName;
        private final long startTime;
        
        AnalysisContext(String algorithmName) {
            this.algorithmName = algorithmName;
            this.startTime = System.nanoTime();
        }
        
        String getAlgorithmName() {
            return algorithmName;
        }
        
        long getExecutionTime() {
            return System.nanoTime() - startTime;
        }
    }
    
    /**
     * 性能统计类 / Performance statistics class
     */
    public static class PerformanceStatistics {
        private final long totalCount;
        private final long totalTime;
        private final double averageTime;
        private final long minTime;
        private final long maxTime;
        private final long errorCount;
        private final double errorRate;
        private final double throughput;
        private final double uptime;
        private final Map<String, AlgorithmStatistics> algorithmStatistics;
        
        public PerformanceStatistics(long totalCount, long totalTime, double averageTime,
                                   long minTime, long maxTime, long errorCount, double errorRate,
                                   double throughput, double uptime, Map<String, AlgorithmStatistics> algorithmStatistics) {
            this.totalCount = totalCount;
            this.totalTime = totalTime;
            this.averageTime = averageTime;
            this.minTime = minTime;
            this.maxTime = maxTime;
            this.errorCount = errorCount;
            this.errorRate = errorRate;
            this.throughput = throughput;
            this.uptime = uptime;
            this.algorithmStatistics = algorithmStatistics;
        }
        
        // Getters
        public long getTotalCount() { return totalCount; }
        public long getTotalTime() { return totalTime; }
        public double getAverageTime() { return averageTime; }
        public long getMinTime() { return minTime; }
        public long getMaxTime() { return maxTime; }
        public long getErrorCount() { return errorCount; }
        public double getErrorRate() { return errorRate; }
        public double getThroughput() { return throughput; }
        public double getUptime() { return uptime; }
        public Map<String, AlgorithmStatistics> getAlgorithmStatistics() { return algorithmStatistics; }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Performance Statistics:\n");
            sb.append(String.format("  Total Analyses: %d\n", totalCount));
            sb.append(String.format("  Total Time: %.2f ms\n", totalTime / 1_000_000.0));
            sb.append(String.format("  Average Time: %.2f ms\n", averageTime / 1_000_000.0));
            sb.append(String.format("  Min Time: %.2f ms\n", minTime / 1_000_000.0));
            sb.append(String.format("  Max Time: %.2f ms\n", maxTime / 1_000_000.0));
            sb.append(String.format("  Error Count: %d\n", errorCount));
            sb.append(String.format("  Error Rate: %.2f%%\n", errorRate));
            sb.append(String.format("  Throughput: %.2f analyses/second\n", throughput));
            sb.append(String.format("  Uptime: %.2f seconds\n", uptime));
            
            if (!algorithmStatistics.isEmpty()) {
                sb.append("  Algorithm Statistics:\n");
                for (Map.Entry<String, AlgorithmStatistics> entry : algorithmStatistics.entrySet()) {
                    sb.append(String.format("    %s: %s\n", entry.getKey(), entry.getValue()));
                }
            }
            
            return sb.toString();
        }
    }
    
    /**
     * 算法统计类 / Algorithm statistics class
     */
    public static class AlgorithmStatistics {
        private final long count;
        private final double averageTime;
        private final long minTime;
        private final long maxTime;
        private final long totalTime;
        
        public AlgorithmStatistics(long count, double averageTime, long minTime, long maxTime, long totalTime) {
            this.count = count;
            this.averageTime = averageTime;
            this.minTime = minTime;
            this.maxTime = maxTime;
            this.totalTime = totalTime;
        }
        
        // Getters
        public long getCount() { return count; }
        public double getAverageTime() { return averageTime; }
        public long getMinTime() { return minTime; }
        public long getMaxTime() { return maxTime; }
        public long getTotalTime() { return totalTime; }
        
        @Override
        public String toString() {
            return String.format("count=%d, avg=%.2fms, min=%.2fms, max=%.2fms",
                count, averageTime / 1_000_000.0, minTime / 1_000_000.0, maxTime / 1_000_000.0);
        }
    }
}