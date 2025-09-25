import java.util.Arrays;

public class ConfidenceCalculationTest {
    
    // Simulate the improved confidence calculation
    private static double calculateImprovedConfidence(double[] chromaFeatures, double bestScore) {
        // 计算基本统计量
        // Calculate basic statistics
        double totalEnergy = 0.0;
        double maxValue = 0.0;
        double minValue = Double.MAX_VALUE;
        
        for (double value : chromaFeatures) {
            totalEnergy += value;
            maxValue = Math.max(maxValue, value);
            minValue = Math.min(minValue, value);
        }
        
        // 检查是否为零向量
        // Check if it's a zero vector
        if (totalEnergy < 1e-10) {
            return 0.0;
        }
        
        // 计算方差来评估分布的非均匀性
        // Calculate variance to assess distribution non-uniformity
        double average = totalEnergy / 12;
        double variance = 0.0;
        
        for (double value : chromaFeatures) {
            variance += (value - average) * (value - average);
        }
        variance /= 12;
        
        // 计算动态范围因子 (0-1)
        // Calculate dynamic range factor (0-1)
        double dynamicRange = maxValue - minValue;
        double rangeFactor = Math.min(1.0, dynamicRange * 2.0); // 对于L2归一化，动态范围通常较小
        
        // 计算方差因子 (0-1)
        // Calculate variance factor (0-1)
        double varianceFactor = Math.min(1.0, variance * 50.0); // 调整方差权重
        
        // 计算峰值突出度
        // Calculate peak prominence
        double[] sortedValues = chromaFeatures.clone();
        Arrays.sort(sortedValues);
        double topThree = sortedValues[11] + sortedValues[10] + sortedValues[9]; // 前三大值
        double bottomNine = totalEnergy - topThree; // 其余九个值的总和
        double peakProminence = bottomNine > 0 ? topThree / bottomNine : 3.0;
        double prominenceFactor = Math.min(1.0, peakProminence / 3.0);
        
        // 综合置信度计算：基于模板匹配分数和特征质量
        // Comprehensive confidence calculation: based on template match score and feature quality
        double featureQuality = (rangeFactor * 0.4 + varianceFactor * 0.4 + prominenceFactor * 0.2);
        double confidence = bestScore * (0.7 + featureQuality * 0.3); // 基础分数70%，特征质量30%
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    private static void printStatistics(String label, double[] chroma, double bestScore, double confidence) {
        double sum = 0.0;
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        
        for (double value : chroma) {
            sum += value;
            max = Math.max(max, value);
            min = Math.min(min, value);
        }
        
        double avg = sum / 12;
        double variance = 0.0;
        for (double value : chroma) {
            variance += (value - avg) * (value - avg);
        }
        variance /= 12;
        
        System.out.println(label);
        System.out.printf("Best Score: %.6f%n", bestScore);
        System.out.printf("Calculated Confidence: %.6f%n", confidence);
        System.out.printf("Statistics: Sum=%.4f, Avg=%.4f, Variance=%.6f, Range=%.4f%n%n", 
                         sum, avg, variance, max - min);
    }
    
    public static void main(String[] args) {
        System.out.println("=== Confidence Calculation Test ===\n");
        
        // Test 1: Current problematic chroma features (too uniform)
        double[] currentChroma = {0.3240, 0.3346, 0.3346, 0.3050, 0.2440, 0.2316, 0.2407, 0.2532, 0.2684, 0.2831, 0.3045, 0.3130};
        double currentBestScore = 0.591219; // E blues score
        double currentConfidence = calculateImprovedConfidence(currentChroma, currentBestScore);
        
        printStatistics("Test 1: Current Chroma Features (too uniform)", currentChroma, currentBestScore, currentConfidence);
        
        // Test 2: Ideal C major chord chroma (enhanced)
        double[] idealChroma = new double[12];
        idealChroma[0] = 0.5;  // C
        idealChroma[4] = 0.4;  // E
        idealChroma[7] = 0.3;  // G
        // Other notes have small values
        for (int i = 0; i < 12; i++) {
            if (i != 0 && i != 4 && i != 7) {
                idealChroma[i] = 0.05;
            }
        }
        
        // L2 normalize
        double norm = 0.0;
        for (double value : idealChroma) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < 12; i++) {
            idealChroma[i] /= norm;
        }
        
        double idealBestScore = 0.8; // Good C major match
        double idealConfidence = calculateImprovedConfidence(idealChroma, idealBestScore);
        
        printStatistics("Test 2: Ideal C Major Chroma (enhanced)", idealChroma, idealBestScore, idealConfidence);
        
        // Test 3: Zero vector
        double[] zeroChroma = new double[12];
        double zeroConfidence = calculateImprovedConfidence(zeroChroma, 0.5);
        
        printStatistics("Test 3: Zero Vector", zeroChroma, 0.5, zeroConfidence);
        
        // Test 4: Perfectly uniform distribution
        double[] uniformChroma = new double[12];
        for (int i = 0; i < 12; i++) {
            uniformChroma[i] = 1.0 / Math.sqrt(12); // L2 normalized uniform
        }
        double uniformConfidence = calculateImprovedConfidence(uniformChroma, 0.5);
        
        printStatistics("Test 4: Perfectly Uniform Distribution", uniformChroma, 0.5, uniformConfidence);
    }
}